import numpy as np

from flx.visualization.det_curve_plotting.DET import DET
from flx.benchmarks.verification import VerificationResult
from flx.benchmarks.identification import IdentificationResult

BBOX_TO_ANCHOR = (1.05, -0.15)


def plot_verification_results(
    path: str,
    results: list[VerificationResult],
    model_labels: list[str],
    plot_title: str,
) -> None:
    det = DET(
        biometric_evaluation_type="algorithm",
        plot_title=plot_title,
        plot_eer_line=True,
    )

    # Customize axes
    det.x_limits = np.array([1e-5, 0.5])
    det.y_limits = np.array([1e-5, 0.5])
    det.x_ticks = np.array([1e-4, 1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
    det.x_ticklabels = np.array(["0.01", "0.1", "1", "5", "20", "40"])
    det.y_ticks = np.array([1e-4, 1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
    det.y_ticklabels = np.array(["0.01", "0.1", "1", "5", "20", "40"])

    # Plot
    det.create_figure()
    for model_label, res in zip(model_labels, results):
        det.plot(
            tar=np.array(res.get_mated_scores()),
            non=np.array(res.get_non_mated_scores()),
            label=model_label,
        )
    det.legend_on(bbox_to_anchor=BBOX_TO_ANCHOR)
    det.save(path, "png")
    det.save(path, "pdf")


def plot_identification_results(
    path: str,
    results: list[IdentificationResult],
    model_labels: list[str],
    plot_title: str,
) -> None:
    det = DET(
        biometric_evaluation_type="identification",
        plot_title=plot_title,
    )

    # Customize axes
    det.x_limits = np.array([1e-5, 0.5])
    det.y_limits = np.array([1e-5, 0.5])
    det.x_ticks = np.array([1e-4, 1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
    det.x_ticklabels = np.array(["0.01", "0.1", "1", "5", "20", "40"])
    det.y_ticks = np.array([1e-4, 1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
    det.y_ticklabels = np.array(["0.01", "0.1", "1", "5", "20", "40"])

    # Plot
    det.create_figure()
    for model_label, res in zip(model_labels, results):
        det.plot(
            tar=np.array(res.get_mated_similarities()),
            non=np.array(res.get_highest_non_mated_similarities()),
            label=model_label,
        )
    det.legend_on(bbox_to_anchor=BBOX_TO_ANCHOR)
    det.save(path, "png")
    det.save(path, "pdf")
