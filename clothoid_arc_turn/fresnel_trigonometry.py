import scipy.special as sc
import math
import numpy as np

def fresnelD(delta: float, lmbda: float = 1, precise: bool = True) -> tuple[float, float]:
    """Cauculate Fresnel-sine and Fresnel-cosine function

    Args:
        delta: half turning angle
        lmbda: turning angle ratio of the clothoid segment
        precise: using scipy to calculate fresnel function or using helper functions

    Returns:
        Fresnel-sine and Fresnel-cosine
    """
    eta = math.sqrt(2 * lmbda * abs(delta) / math.pi)
    if eta == 0:
        return (math.sin(delta), math.cos(delta))
    ds = math.sin(delta)
    dc = math.cos(delta)

    if precise:
        fs, fc = sc.fresnel(eta)
        if delta < 0:
            fs *= -1
        return(
            (-dc * fs + ds * fc) / eta,
            (dc * fc + ds * fs) / eta
        )
    else:
        helper_func_f = (1 + 0.926 * eta) / (2 + 1.792 * eta + 3.104 * eta**2)
        if delta < 0:
            helper_func_f *= -1
        helper_func_g = 1.0 / (2 + 4.142 * eta + 3.492 * eta**2 + 6.670 * eta**3)
        dss = math.sin((1 - lmbda) * delta)
        dsc = math.cos((1 - lmbda) * delta)

        product_s_ex = ds - dc if delta > 0 else ds + dc
        product_s = helper_func_f * dsc- helper_func_g * dss + 0.5 * product_s_ex

        product_c_ex = dc + ds if delta > 0 else dc - ds
        product_c = -helper_func_f * dss - helper_func_g * dsc + 0.5 * product_c_ex

        return (
            product_s / eta,
            product_c / eta
        )
def fresnelDe(delta: float, lmbda: float = 1, precise: bool = True) -> tuple[float, float]:
    """Cauculate extended Fresnel-sine and Fresnel-cosine function

    Args:
        delta: half turning angle
        lmbda: turning angle ratio of the clothoid segment

    Returns:
        Extended Fresnel-sine and Fresnel-cosine
    """
    fds, fdc = fresnelD(delta, lmbda, precise)
    extra_ce = math.sin((1 - lmbda) * delta)
    extra_se = 1 - math.cos((1 - lmbda) * delta)
    fdce = 2 * lmbda * delta * fdc + extra_ce
    fdse = 2 * lmbda * delta * fds + extra_se
    return fdse, fdce
def fresnelDAndDe(delta: float, lmbda: float = 1) -> tuple[float, float, float, float]:
    """Cauculate extended Fresnel-cosine and Fresnel-sine function

    Args:
        delta: half turning angle
        lmbda: turning angle ratio of the clothoid segment

    Returns:
        Fresnel-cosine, Fresnel-sine, Extended Fresnel-cosine, Extended Fresnel-sine
    """
    fds, fdc = fresnelD(delta, lmbda)
    extra_ce = math.sin((1 - lmbda) * delta)
    extra_se = 1 - math.cos((1 - lmbda) * delta)
    fdce = 2 * lmbda * delta * fdc + extra_ce
    fdse = 2 * lmbda * delta * fds + extra_se

    return fds, fdc, fdse, fdce
def fresnelDeWithDerivatives(delta: float, lmbda: float = 1) -> tuple[float, float, float, float]:
    fds, fdc, fdse, fdce = fresnelDAndDe(delta, lmbda)
    return fdse, fdce, delta * fds, delta * fdc

def fresnelDeWith2ndDerivatives(delta: float, lmbda: float = 1) -> tuple[float, float, float, float, float, float]:
    fds, fdc, fdse, fdce = fresnelDAndDe(delta, lmbda)
    fddce = delta * (1 - fdc - fdse + 2 * fds * delta * lmbda) / (2 * lmbda)
    fddse = -delta * (-fdce + fds + 2 * fdc * delta * lmbda) / (2 * lmbda)
    return fdse, fdce, delta * fds, delta * fdc, fddse, fddce

def inverseFresnelDeCNewton(delta: float, target_fdce: float, halley: bool = False) -> float:
    current_lmbda = 0.5
    lmbda_change = 1.0
    # SOLVE FC(delta;lmbda) - target_fdce = 0
    while abs(lmbda_change) > 1e-3:
        # _, fdce, _, d_fdce = fresnelDeWithDerivatives(delta, current_lmbda)
        # new_lmbda = current_lmbda - (fdce - target_fdce) / d_fdce

        # FOR BENCHMARK (remove unusable variables)
        eta = math.sqrt(2 * current_lmbda * abs(delta) / math.pi)
        ds = math.sin(delta)
        dc = math.cos(delta)
        if eta == 0:
            fdc = dc
        else:
            fs, fc = sc.fresnel(eta)
            if delta < 0:
                fs *= -1
            fdc = (dc * fc + ds * fs) / eta
        extra_ce = math.sin((1 - current_lmbda) * delta)

        d_fdce = delta * fdc
        new_lmbda = -current_lmbda - (extra_ce-target_fdce) / d_fdce
        # end FOR BENCHMARK

        if new_lmbda > 1:
            new_lmbda = 1
        if new_lmbda < 0:
            new_lmbda = 0
        lmbda_change = new_lmbda - current_lmbda
        current_lmbda = new_lmbda
    return current_lmbda
def inverseFresnelDeSNewton(delta: float, target_fdse: float) -> float:
    current_lmbda = 0.5
    lmbda_change = 1.0
    # SOLVE FS(delta;lmbda) - target_fdse = 0
    while abs(lmbda_change) > 1e-3:
        fdse, _, d_fdse, _ = fresnelDeWithDerivatives(delta, current_lmbda)
        if d_fdse == 0:
            break
        new_lmbda = current_lmbda - (fdse - target_fdse) / d_fdse
        if new_lmbda > 1:
            new_lmbda = 1
        if new_lmbda < 0:
            new_lmbda = 0
        lmbda_change = new_lmbda - current_lmbda
        current_lmbda = new_lmbda
    return current_lmbda

def inverseFresnelDeTBisection(delta: float, target_fdte: float, lmbda_min: float, lmbda_max: float) -> float:
    fdse_min, fdce_min = fresnelDe(delta, lmbda_min)
    fdte_min = fdse_min / fdce_min
    fdse_max, fdce_max = fresnelDe(delta, lmbda_max)
    fdte_max = fdse_max / fdce_max

    while lmbda_max - lmbda_min > 1e-3:
        # less than 6 iter
        lmbda_mid = (lmbda_min + lmbda_max) / 2
        fdse_mid, fdce_mid = fresnelDe(delta, lmbda_mid)
        fdte_mid = fdse_mid / fdce_mid
        if (fdte_mid - target_fdte) * (fdte_min - target_fdte) > 0:
            # min and mid are on same side
            # using mid as new min
            lmbda_min = lmbda_mid
            fdte_min = lmbda_mid
        else:
            # using mid as new max
            lmbda_max = lmbda_mid
            fdte_max = lmbda_mid
    return (lmbda_min + lmbda_max) / 2
def inverseFresnelDeTNewton(delta: float, target_fdte: float) -> float:
    current_lmbda = 0.5
    # fdte change its sign somewhere between lmbda in [0,1]
    fdse0, fdce0 = fresnelDe(delta, 0)
    if fdse0 / fdce0 * target_fdte < 0:
        # different sign at lmbda = 0, finding at side lmbda = 1
        direction = 0.25
    else:
        # same sign at lmbda = 0, finding at side lmbda = 0
        direction = -0.25

    while True:
        fdse_current, fdce_current = fresnelDe(delta, current_lmbda)
        fdte_current = fdse_current / fdce_current
        if fdte_current * target_fdte > 0:
            break
        current_lmbda += direction
        direction /= 2

    lmbda_change = 1.0
    test_cnt = 0

    refresh_zero = False

    # SOLVE FS(delta;lmbda) / FC(delta;lmbda) - target_fdte = 0
    while abs(lmbda_change) > 1e-3:
        fdse, fdce, d_fdse, d_fdce = fresnelDeWithDerivatives(delta, current_lmbda)

        fdte = fdse / fdce
        d_fdte = (fdce * d_fdse - fdse * d_fdce) / (fdce ** 2)

        if fdte * target_fdte < 0:
            # DO NOT let the function cross the zero line
            lmbda_change /= 2
            current_lmbda -= lmbda_change
            continue

        if d_fdte == 0:
            if current_lmbda == 1 and abs(fdte) > abs(target_fdte):
                if refresh_zero:
                    return inverseFresnelDeTBisection(delta, target_fdte, 0.95, 1.00)
                else:
                    refresh_zero = True
                    current_lmbda = 0.95
                    continue

        new_lmbda = current_lmbda - (fdte - target_fdte) / d_fdte
        if new_lmbda >= 1:
            new_lmbda = 1
        if new_lmbda <= 0:
            new_lmbda = 0
        lmbda_change = new_lmbda - current_lmbda
        current_lmbda = new_lmbda

        test_cnt += 1

        if test_cnt > 20:
            print(f'err!, {delta}, {target_fdte}')
            break
    return current_lmbda

def fresnelMcDsWithLmbdaAndDeltaDeltaDerivatives(delta: float, delta_delta: float, lmbda: float) -> tuple[float,float,float,float,float,float]:
    """Solve mC and dS with their derivatives

    mC(delta,delta_delta,lmbda) := 0.5 * (C(delta+delta_delta, lmbda) + C(delta-delta_delta, lmbda))
    dS(delta,delta_delta,lmbda) := 0.5 * (S(delta+delta_delta, lmbda) - S(delta-delta_delta, lmbda))

    Args:
        delta: half turning angle
        delta_delta: difference between two turning angles
        lmbda: turning angle ratio of the clothoid segment

    Returns:
        mC, dS,
        d(mC)/d(lmbda),
        d(dS)/d(lmbda),
        d(mC)/d(delta_delta),
        d(dS)/d(delta_delta)
    """

    delta_0 = delta + delta_delta
    delta_1 = delta - delta_delta
    fds0, fdc0, fdse0, fdce0 = fresnelDAndDe(delta_0, lmbda)
    fds1, fdc1, fdse1, fdce1 = fresnelDAndDe(delta_1, lmbda)
    mC = 0.5 * (fdce0 + fdce1)
    dS = 0.5 * (fdse0 - fdse1)

    fdce0_lmbda = delta_0 * fdc0
    fdce1_lmbda = delta_1 * fdc1
    fdse0_lmbda = delta_0 * fds0
    fdse1_lmbda = delta_1 * fds1

    fdce0_delta =   -fdse0 + 1 + fdc0 * lmbda
    fdce1_delta = -(-fdse1 + 1 + fdc1 * lmbda)
    fdse0_delta =   fdce0 + fds0 * lmbda
    fdse1_delta = -(fdce1 + fds1 * lmbda)

    mC_lmbda = 0.5 * (fdce0_lmbda + fdce1_lmbda)
    dS_lmbda = 0.5 * (fdse0_lmbda - fdse1_lmbda)
    mC_delta = 0.5 * (fdce0_delta + fdce1_delta)
    dS_delta = 0.5 * (fdse0_delta - fdse1_delta)

    return mC, dS, mC_lmbda, dS_lmbda, mC_delta, dS_delta

def fresnelMcDsWithDeltaDeltaDerivatives2(delta: float, delta_delta: float, lmbda: float) -> tuple[float,float,float,float,float,float]:
    """Solve mC and dS with their derivatives

    mC(delta,delta_delta,lmbda) := 0.5 * (C(delta+delta_delta, lmbda) + C(delta-delta_delta, lmbda))
    dS(delta,delta_delta,lmbda) := 0.5 * (S(delta+delta_delta, lmbda) - S(delta-delta_delta, lmbda))

    Args:
        delta: half turning angle
        delta_delta: difference between two turning angles
        lmbda: turning angle ratio of the clothoid segment

    Returns:
        mC, dS,
        d(mC)/d(delta_delta),
        d(dS)/d(delta_delta)
    """

    delta_0 = delta + delta_delta
    delta_1 = delta - delta_delta
    fds0, fdc0, fdse0, fdce0 = fresnelDAndDe(delta_0, lmbda)
    fds1, fdc1, fdse1, fdce1 = fresnelDAndDe(delta_1, lmbda)
    mC = 0.5 * (fdce0 + fdce1)
    dS = 0.5 * (fdse0 - fdse1)

    fdce0_delta =   -fdse0 + 1 + fdc0 * lmbda
    fdce1_delta = -(-fdse1 + 1 + fdc1 * lmbda)
    fdse0_delta =   fdce0 + fds0 * lmbda
    fdse1_delta = -(fdce1 + fds1 * lmbda)

    fdc0_delta = (fdce0_delta - (1 - lmbda) * (1 - fdse0 + 2 * fds0 * delta_0 * lmbda)) / (2 * lmbda)
    fdc1_delta = (fdce1_delta - (1 - lmbda) * (1 - fdse1 + 2 * fds1 * delta_1 * lmbda)) / (2 * lmbda)
    fds0_delta = (fdse0_delta - (1 - lmbda) * (fdce0 - 2 * fdc0 * delta_0 * lmbda)) / (2 * lmbda)
    fds1_delta = (fdse1_delta - (1 - lmbda) * (fdce1 - 2 * fdc1 * delta_1 * lmbda)) / (2 * lmbda)

    fdce0_delta2 =   -fdse0_delta + fdc0_delta * lmbda
    fdce1_delta2 = -(-fdse1_delta + fdc1_delta * lmbda)
    fdse0_delta2 =    fdce0_delta + fds0_delta * lmbda
    fdse1_delta2 =  -(fdce1_delta + fds1_delta * lmbda)

    mC_delta = 0.5 * (fdce0_delta + fdce1_delta)
    dS_delta = 0.5 * (fdse0_delta - fdse1_delta)

    mC_delta2 = 0.5 * (fdce0_delta2 + fdce1_delta2)
    dS_delta2 = 0.5 * (fdse0_delta2 - fdse1_delta2)

    return mC, dS, mC_delta, dS_delta, mC_delta2, dS_delta2
