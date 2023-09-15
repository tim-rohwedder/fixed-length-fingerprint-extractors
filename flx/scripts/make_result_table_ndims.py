from flx.benchmarks.verification import VerificationResult
from flx.benchmarks.identification import IdentificationResult
from flx.setup.experiments import (
    get_experiments,
    get_reweighting_experiments,
    Experiment,
    DatasetLoader,
)

PERC_SYM = ""


def _get_experiments_with_rw(
    testset_keys: list[str], extractor_keys: list[str]
) -> list[Experiment]:
    _, _, experiments = get_experiments(
        testset_keys=testset_keys, extractor_keys=extractor_keys
    )
    rw_experiments = get_reweighting_experiments(experiments)
    return list(experiments.values()) + list(rw_experiments.values())


def _get_fnmr_at_fmr_in_percent(exp: Experiment, fmr: float) -> float:
    result = exp.load_verification_benchmark_results()
    t = result.threshold_for_fmr(fmr / 100)
    return result.false_non_match_rate([t])[0] * 100


def _get_eer_in_percent(exp: Experiment) -> float:
    result = exp.load_verification_benchmark_results()
    return result.get_equal_error_rate() * 100


def _format_multirow(nrows, val) -> str:
    return "\\multirow{" + str(nrows) + "}{*}{" + str(val) + "}"


def _underline_if(val, cond: bool) -> str:
    fval = f"{val:.2f}"
    if cond:
        return "\\underline{" + fval + "}"
    return fval


def _get_eer_percent(exp: Experiment) -> float:
    res = exp.load_verification_benchmark_results()
    return res.get_equal_error_rate() * 100


def _make_section(name: str, nrows: int) -> str:
    outstr = "\n\\midrule\n"
    outstr += "\\multirow{" + str(nrows) + "}{*}{ \\textbf{" + name + " } }\n"
    return outstr


def _make_row(
    dims: int, ops: int, fnmr: float, fnmr_rw: float, eer: float, eer_rw: float
) -> str:
    outstr = f"& {dims} & {ops} & "
    outstr += f"{_underline_if(fnmr, fnmr < fnmr_rw)} {PERC_SYM} & "
    outstr += f"{_underline_if(fnmr_rw, fnmr > fnmr_rw)} {PERC_SYM} & "
    outstr += f"{_underline_if(eer, eer < eer_rw)} {PERC_SYM} & "
    outstr += f"{_underline_if(eer_rw, eer > eer_rw)} {PERC_SYM} "
    outstr += "\\\\ \n"
    return outstr


def main():
    # Table for different preprocessing methods
    EXTRACTOR_KEYS = [
        "DeepPrint_Tex_32",
        "DeepPrint_Tex_64",
        "DeepPrint_Tex_128",
        "DeepPrint_Tex_256",
        "DeepPrint_Tex_512",
        "DeepPrint_Tex_1024",
        "DeepPrint_Tex_2048",
    ]

    # Num. Dimensions, Num. Operations, FNMR@0.1%, FNMR@0.1% (rw), EER, EER (rw)

    outstr_opt = _make_section("Optical Database", len(EXTRACTOR_KEYS))
    outstr_cap = _make_section("Capacitive Database", len(EXTRACTOR_KEYS))
    for e in EXTRACTOR_KEYS:
        optical, optical_rw = _get_experiments_with_rw(["mcyt330_optical"], [e])
        capacitive, capacitive_rw = _get_experiments_with_rw(
            ["mcyt330_capacitive"], [e]
        )

        dims = int(e.split("_")[-1])
        ops = 2 * dims - 1

        opt_fnmr = _get_fnmr_at_fmr_in_percent(optical, 0.1)
        cap_fnmr = _get_fnmr_at_fmr_in_percent(capacitive, 0.1)
        opt_rw_fnmr = _get_fnmr_at_fmr_in_percent(optical_rw, 0.1)
        cap_rw_fnmr = _get_fnmr_at_fmr_in_percent(capacitive_rw, 0.1)

        opt_eer = _get_eer_in_percent(optical)
        cap_eer = _get_eer_in_percent(capacitive)
        opt_rw_eer = _get_eer_in_percent(optical_rw)
        cap_rw_eer = _get_eer_in_percent(capacitive_rw)

        outstr_opt += _make_row(dims, ops, opt_fnmr, opt_rw_fnmr, opt_eer, opt_rw_eer)
        outstr_cap += _make_row(dims, ops, cap_fnmr, cap_rw_fnmr, cap_eer, cap_rw_eer)

    with open("table.tex", "w") as f:
        f.write(outstr_opt + "\n\n" + outstr_cap)


if __name__ == "__main__":
    main()
