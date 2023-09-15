from flx.benchmarks.verification import VerificationResult
from flx.benchmarks.identification import IdentificationResult
from flx.setup.experiments import (
    get_experiments,
    get_reweighting_experiments,
    Experiment,
    DatasetLoader,
)


def _get_experiments_with_rw(
    testset_keys: list[str], extractor_keys: list[str]
) -> list[Experiment]:
    _, _, experiments = get_experiments(
        testset_keys=testset_keys, extractor_keys=extractor_keys
    )
    rw_experiments = get_reweighting_experiments(experiments)
    return list(experiments.values()) + list(rw_experiments.values())


def _get_rank1_identification_rate(exp: Experiment) -> float:
    result = exp.load_closed_set_benchmark_results()
    all_ranks = result.get_mated_ranks()
    return (sum([1 if r == 1 else 0 for r in all_ranks]) / len(all_ranks)) * 100


def _get_fnir_at_fpir_in_percent(exp: Experiment, fpir: float) -> float:
    result: IdentificationResult = exp.load_open_set_benchmark_results()
    return result.false_negative_identification_rate(fpir=fpir / 100) * 100


def _format_multirow(nrows, val) -> str:
    return "\\multirow{" + str(nrows) + "}{*}{" + str(val) + "}"


def _underline_if(val, cond: bool) -> str:
    fval = f"{val:.2f}"
    if cond:
        return "\\underline{" + fval + "}"
    return fval


def main():
    # Table for different preprocessing methods
    EXTRACTOR_KEYS = [
        "DeepPrint_Tex_512",
        "DeepPrint_Minu_512",
        "DeepPrint_TexMinu_512",
    ]

    LABELS = ["texture branch", "minutia branch", "texture and minutia branch"]
    # Num. Dimensions, Num. Operations, Reweighted, EER (capacitive), EER (optical)

    outstr = ""
    for e, label in zip(EXTRACTOR_KEYS, LABELS):
        optical, optical_rw = _get_experiments_with_rw(["mcyt330_optical"], [e])
        capacitive, capacitive_rw = _get_experiments_with_rw(
            ["mcyt330_capacitive"], [e]
        )

        opt_rank1 = _get_rank1_identification_rate(optical)
        opt_fnir = _get_fnir_at_fpir_in_percent(optical, 0.1)

        cap_rank1 = _get_rank1_identification_rate(capacitive)
        cap_fnir = _get_fnir_at_fpir_in_percent(capacitive, 0.1)

        opt_rw_rank1 = _get_rank1_identification_rate(optical_rw)
        opt_rw_fnir = _get_fnir_at_fpir_in_percent(optical_rw, 0.1)

        cap_rw_rank1 = _get_rank1_identification_rate(capacitive_rw)
        cap_rw_fnir = _get_fnir_at_fpir_in_percent(capacitive_rw, 0.1)

        dims = 512
        outstr += "\\midrule \n"
        outstr += (
            "\\multirow{2}{*}{"
            + str(label)
            + "} & \\multirow{2}{*}{"
            + str(dims)
            + "} & No &"
        )
        outstr += f"{_underline_if(opt_rank1, opt_rank1 > opt_rw_rank1)} \\% & "
        outstr += f"{_underline_if(opt_fnir, opt_fnir < opt_rw_fnir)} \\% & "
        outstr += f"{_underline_if(cap_rank1, cap_rank1 > cap_rw_rank1)} \\% & "
        outstr += f"{_underline_if(cap_fnir, cap_fnir < cap_rw_fnir)} \\% "
        outstr += "\\\\ \n"
        outstr += " & & Yes & "
        outstr += f"{_underline_if(opt_rw_rank1, opt_rank1 < opt_rw_rank1)} \\% & "
        outstr += f"{_underline_if(opt_rw_fnir, opt_fnir > opt_rw_fnir)} \\% & "
        outstr += f"{_underline_if(cap_rw_rank1, cap_rank1 < cap_rw_rank1)} \\% & "
        outstr += f"{_underline_if(cap_rw_fnir, cap_fnir > cap_rw_fnir)} \\% "
        outstr += "\\\\ \n"

    with open("table.tex", "w") as f:
        f.write(outstr)


if __name__ == "__main__":
    main()
