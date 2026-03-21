# core/fusion/vio_covariance_error.py
# MicroMind — S-NEP-04 Step 04-A


class VIOCovarianceError(ValueError):
    """
    Raised when a VIO measurement covariance is degenerate.

    Degenerate means any diagonal element of the 3×3 position covariance
    block is zero or negative.  A degenerate covariance would produce an
    ill-conditioned Kalman gain and must never be fused.

    Corresponds to Integration Failure Mode IFM-04.
    """
