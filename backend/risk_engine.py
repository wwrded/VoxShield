"""
Risk scoring: turns a model probability into an operational decision.

The classifier answers a narrow question - "how synthetic does this audio
look?" - on a 0..1 scale. An organisation needs something different: a tier, a
named action, and a statement of which security property is at stake. This
module performs that translation, and it is deliberately independent of any
single industry.

Deployment profiles
-------------------
The same detector protects very different processes, and they do not share a
risk appetite. A bank authorising a wire transfer would rather stop a genuine
customer than release funds to a clone, so it wants a low spoof threshold and
tolerates false alarms. A high-volume support desk answering password-reset
calls cannot afford to challenge one caller in twenty, so it wants a higher
threshold. Encoding that as a profile - rather than hard-coding one industry's
thresholds - is what makes the system deployable outside a contact centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DeploymentProfile:
    """Risk appetite and vocabulary for one class of deployment."""

    key: str
    display_name: str
    description: str
    # Fractions of the calibrated operating point. 0.6 means "flag as
    # suspicious at 60% of the score that the model calls a decision".
    review_ratio: float
    block_ratio: float
    asset_at_risk: str
    step_up_action: str
    block_action: str


PROFILES: Dict[str, DeploymentProfile] = {
    "generic": DeploymentProfile(
        key="generic",
        display_name="General purpose",
        description=(
            "Default balanced profile for any organisation that accepts voice "
            "as an input channel and has no stricter regulatory driver."
        ),
        review_ratio=0.60,
        block_ratio=1.00,
        asset_at_risk="account data and privileged actions",
        step_up_action="Request a second, non-voice factor before continuing.",
        block_action="Stop the transaction and route the session to a human reviewer.",
    ),
    "contact_center": DeploymentProfile(
        key="contact_center",
        display_name="Contact centre / customer support",
        description=(
            "Inbound agent desktops and IVR front-ends. High call volume, so "
            "false alarms carry a real handling-time cost."
        ),
        review_ratio=0.70,
        block_ratio=1.10,
        asset_at_risk="customer PII and account servicing actions",
        step_up_action=(
            "Run step-up verification: out-of-band OTP, or knowledge factors "
            "that are not recoverable from public data."
        ),
        block_action=(
            "Do not disclose account details or execute value transactions. "
            "Escalate to fraud operations and retain the recording."
        ),
    ),
    "financial_services": DeploymentProfile(
        key="financial_services",
        display_name="Banking / payments authorisation",
        description=(
            "Voice-authorised transfers, card servicing and treasury "
            "instructions. Loss per missed detection is large, so the profile "
            "is deliberately conservative."
        ),
        review_ratio=0.40,
        block_ratio=0.75,
        asset_at_risk="funds, payment instructions and credit facilities",
        step_up_action=(
            "Hold the instruction and verify through an enrolled second "
            "channel before release."
        ),
        block_action=(
            "Refuse the instruction, freeze the request and raise a suspicious "
            "activity report per internal AML/fraud policy."
        ),
    ),
    "enterprise_helpdesk": DeploymentProfile(
        key="enterprise_helpdesk",
        display_name="Internal IT service desk",
        description=(
            "Protects against the credential-reset and MFA-reset social "
            "engineering path that voice cloning has made cheap."
        ),
        review_ratio=0.50,
        block_ratio=0.85,
        asset_at_risk="employee credentials, MFA enrolment and remote access",
        step_up_action=(
            "Do not reset credentials or MFA factors on this call. Verify "
            "through the enrolled device or the caller's line manager."
        ),
        block_action=(
            "Deny the reset request, preserve the recording and notify the "
            "security operations centre of an attempted identity attack."
        ),
    ),
    "kyc_onboarding": DeploymentProfile(
        key="kyc_onboarding",
        display_name="Remote identity proofing / KYC",
        description=(
            "Voice or video onboarding where a synthetic sample would create a "
            "durable fraudulent identity rather than a single fraudulent call."
        ),
        review_ratio=0.35,
        block_ratio=0.70,
        asset_at_risk="identity records and downstream account provisioning",
        step_up_action="Route to manual identity review before provisioning.",
        block_action=(
            "Reject the enrolment attempt and retain the artefacts as evidence "
            "of a presentation attack."
        ),
    ),
    "media_verification": DeploymentProfile(
        key="media_verification",
        display_name="Media / evidence authentication",
        description=(
            "Newsroom, legal or HR review of a recording whose provenance is "
            "disputed. Reports evidence, and never blocks anything."
        ),
        review_ratio=0.50,
        block_ratio=0.90,
        asset_at_risk="the evidential integrity of the recording",
        step_up_action=(
            "Treat as unverified. Seek the original file and its chain of "
            "custody before relying on the content."
        ),
        block_action=(
            "Strong indication of synthesis. Do not publish or rely on this "
            "recording without independent corroboration."
        ),
    ),
}

DEFAULT_PROFILE = "generic"


@dataclass
class RiskAssessment:
    """Structured result returned to the caller."""

    risk_score: float
    risk_tier: str
    classification_label: str
    verdict: str
    color: str
    spoof_probability: float
    decision_threshold: float
    margin: float
    profile: str
    profile_display_name: str
    action_code: str
    recommendation: str
    cia_impact: Dict[str, str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    caveats: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier,
            "classification_label": self.classification_label,
            "verdict": self.verdict,
            "color": self.color,
            "spoof_probability": self.spoof_probability,
            "decision_threshold": self.decision_threshold,
            "margin": self.margin,
            "profile": self.profile,
            "profile_display_name": self.profile_display_name,
            "action_code": self.action_code,
            "recommendation": self.recommendation,
            "cia_impact": self.cia_impact,
            "evidence": self.evidence,
            "caveats": self.caveats,
        }


class RiskScoringEngine:
    """Maps a spoof probability onto a tiered, profile-aware decision."""

    LOW_COLOR = "#10b981"
    MEDIUM_COLOR = "#f59e0b"
    HIGH_COLOR = "#ef4444"

    @staticmethod
    def resolve_profile(profile_key: Optional[str]) -> DeploymentProfile:
        return PROFILES.get((profile_key or DEFAULT_PROFILE).lower(),
                            PROFILES[DEFAULT_PROFILE])

    @classmethod
    def assess(
        cls,
        spoof_probability: float,
        decision_threshold: float,
        profile_key: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
        caveats: Optional[list] = None,
    ) -> RiskAssessment:
        """
        Parameters
        ----------
        spoof_probability
            Model output in [0, 1]; 1 means "certainly synthetic".
        decision_threshold
            The calibrated operating point recorded at training time (the
            Equal Error Rate threshold), not a hard-coded 0.5. Using the
            trained threshold is what keeps the tiers meaningful when the
            score distribution is skewed.
        profile_key
            Deployment profile; see PROFILES.
        evidence
            Interpretable per-clip measurements shown alongside the score.
        caveats
            Statements that must travel with the result, such as the model
            being uncalibrated.
        """
        profile = cls.resolve_profile(profile_key)
        prob = max(0.0, min(1.0, float(spoof_probability)))
        threshold = max(1e-3, min(0.999, float(decision_threshold)))

        review_at = min(0.99, threshold * profile.review_ratio)
        block_at = min(0.99, threshold * profile.block_ratio)
        if block_at <= review_at:
            block_at = min(0.99, review_at + 0.05)

        if prob < review_at:
            tier, label = "LOW", "BONAFIDE_LIKELY"
            verdict = "No synthetic-speech evidence found"
            color = cls.LOW_COLOR
            action_code = "PROCEED"
            recommendation = (
                "Continue with the organisation's normal identity checks. This "
                "result is one signal and does not by itself authenticate the "
                "speaker."
            )
            cia_impact = {
                "confidentiality": (
                    f"No detector-driven concern for {profile.asset_at_risk}."
                ),
                "integrity": "Acoustic evidence is consistent with a live human talker.",
                "availability": "No service impact; normal routing.",
            }
        elif prob < block_at:
            tier, label = "MEDIUM", "INCONCLUSIVE_SUSPICIOUS"
            verdict = "Partially synthetic-consistent acoustics"
            color = cls.MEDIUM_COLOR
            action_code = "STEP_UP_VERIFICATION"
            recommendation = profile.step_up_action
            cia_impact = {
                "confidentiality": (
                    f"Elevated risk to {profile.asset_at_risk}; impersonation "
                    "cannot be ruled out."
                ),
                "integrity": (
                    "Some spectral characteristics resemble vocoder output, but "
                    "channel noise and codec loss produce similar effects."
                ),
                "availability": "Hold irreversible actions pending verification.",
            }
        else:
            tier, label = "HIGH", "SYNTHETIC_SPEECH_LIKELY"
            verdict = "Strong synthetic-speech evidence"
            color = cls.HIGH_COLOR
            action_code = "BLOCK_AND_ESCALATE"
            recommendation = profile.block_action
            cia_impact = {
                "confidentiality": (
                    f"Critical risk to {profile.asset_at_risk} through "
                    "impersonation of an authorised speaker."
                ),
                "integrity": (
                    "Spectral evidence is consistent with machine-generated or "
                    "voice-converted speech."
                ),
                "availability": (
                    "Contain the session and preserve the recording as evidence."
                ),
            }

        # Distance from the operating point, expressed in [0, 1]. This is a
        # margin, not a statistical confidence interval, and it is named as
        # such so that nobody reads it as a calibrated certainty.
        span = max(threshold, 1.0 - threshold)
        margin = round(min(1.0, abs(prob - threshold) / span) * 100, 1)

        return RiskAssessment(
            risk_score=round(prob * 100, 1),
            risk_tier=tier,
            classification_label=label,
            verdict=verdict,
            color=color,
            spoof_probability=round(prob, 4),
            decision_threshold=round(threshold, 4),
            margin=margin,
            profile=profile.key,
            profile_display_name=profile.display_name,
            action_code=action_code,
            recommendation=recommendation,
            cia_impact=cia_impact,
            evidence=evidence or {},
            caveats=list(caveats or []),
        )


def list_profiles() -> list:
    """Profile catalogue for the dashboard selector."""
    return [
        {
            "key": p.key,
            "display_name": p.display_name,
            "description": p.description,
            "asset_at_risk": p.asset_at_risk,
        }
        for p in PROFILES.values()
    ]
