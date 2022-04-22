import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class NotEnoughDataError(Exception):
    """unable to collect enough swing data to initialize strategy"""


def average_true_range(
        df: pd.DataFrame,
        window: int,
        _h: str = 'high',
        _l: str = 'low',
        _c: str = 'close',
):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    """
    _max = df[_h].combine(df[_c].shift(), max)
    _min = df[_l].combine(df[_c].shift(), min)
    atr = (_max - _min).rolling(window=window).mean()
    # atr = (_max - _min).ewm(span=window, min_periods=window).mean()
    return atr


def lower_upper_ohlc(df, is_relative=False):
    if is_relative == True:
        rel = "r"
    else:
        rel = ""
    if "Open" in df.columns:
        ohlc = [rel + "Open", rel + "High", rel + "Low", rel + "Close"]
    elif "open" in df.columns:
        ohlc = [rel + "open", rel + "high", rel + "low", rel + "close"]

    try:
        _o, _h, _l, _c = [ohlc[h] for h in range(len(ohlc))]
    except:
        _o = _h = _l = _c = np.nan
    return _o, _h, _l, _c


def regime_args(df, lvl, is_relative=False):
    if ("Low" in df.columns) & (is_relative == False):
        reg_val = [
            "Lo1",
            "Hi1",
            "Lo" + str(lvl),
            "Hi" + str(lvl),
            "rg",
            "clg",
            "flr",
            "rg_ch",
        ]
    elif ("low" in df.columns) & (is_relative == False):
        reg_val = [
            "lo1",
            "hi1",
            "lo" + str(lvl),
            "hi" + str(lvl),
            "rg",
            "clg",
            "flr",
            "rg_ch",
        ]
    elif ("Low" in df.columns) & (is_relative == True):
        reg_val = [
            "rL1",
            "rH1",
            "rL" + str(lvl),
            "rH" + str(lvl),
            "rrg",
            "rclg",
            "rflr",
            "rrg_ch",
        ]
    elif ("low" in df.columns) & (is_relative == True):
        reg_val = [
            "rl1",
            "rh1",
            "rl" + str(lvl),
            "rh" + str(lvl),
            "rrg",
            "rclg",
            "rflr",
            "rrg_ch",
        ]

    try:
        rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch = [
            reg_val[s] for s in range(len(reg_val))
        ]
    except:
        rt_lo = rt_hi = slo = shi = rg = clg = flr = rg_ch = np.nan
    return rt_lo, rt_hi, slo, shi, rg, clg, flr, rg_ch


def regime_breakout(df, _h, _l, window):
    hl = np.where(
        df[_h] == df[_h].rolling(window).max(),
        1,
        np.where(df[_l] == df[_l].rolling(window).min(), -1, np.nan),
    )
    roll_hl = pd.Series(index=df.index, data=hl).fillna(method="ffill")
    return roll_hl


def hilo_alternation(hilo, dist=None, hurdle=None):
    i = 0
    while (
            np.sign(hilo.shift(1)) == np.sign(hilo)
    ).any():  # runs until duplicates are eliminated

        # removes swing lows > swing highs
        hilo.loc[
            (np.sign(hilo.shift(1)) != np.sign(hilo))
            & (hilo.shift(1) < 0)  # hilo alternation test
            & (np.abs(hilo.shift(1)) < np.abs(hilo))  # previous datapoint:  high
            ] = np.nan  # high[-1] < low, eliminate low

        hilo.loc[
            (np.sign(hilo.shift(1)) != np.sign(hilo))
            & (hilo.shift(1) > 0)  # hilo alternation
            & (np.abs(hilo) < hilo.shift(1))  # previous swing: low
            ] = np.nan  # swing high < swing low[-1]

        # alternation test: removes duplicate swings & keep extremes
        hilo.loc[
            (np.sign(hilo.shift(1)) == np.sign(hilo))
            & (hilo.shift(1) < hilo)  # same sign
            ] = np.nan  # keep lower one

        hilo.loc[
            (np.sign(hilo.shift(-1)) == np.sign(hilo))
            & (hilo.shift(-1) < hilo)  # same sign, forward looking
            ] = np.nan  # keep forward one

        # removes noisy swings: distance test
        if pd.notnull(dist):
            hilo.loc[
                (np.sign(hilo.shift(1)) != np.sign(hilo))
                & (np.abs(hilo + hilo.shift(1)).div(dist, fill_value=1) < hurdle)
                ] = np.nan

        # reduce hilo after each pass
        hilo = hilo.dropna().copy()
        i += 1
        if i == 4:  # breaks infinite loop
            break
        return hilo


def historical_swings(df, _o='open', _h='high', _l='low', _c='close', round_place=2, lvl_limit=3):
    reduction = df[[_o, _h, _l, _c]].copy()

    reduction["avg_px"] = round(reduction[[_h, _l, _c]].mean(axis=1), round_place)
    highs = reduction["avg_px"].values
    lows = -reduction["avg_px"].values
    reduction_target = len(reduction) // 100
    #     print(reduction_target )

    n = 0
    while len(reduction) >= reduction_target:
        highs_list = find_peaks(highs, distance=1, width=0)
        lows_list = find_peaks(lows, distance=1, width=0)
        hilo = reduction.iloc[lows_list[0]][_l].sub(
            reduction.iloc[highs_list[0]][_h], fill_value=0
        )

        # Reduction dataframe and alternation loop
        hilo_alternation(hilo, dist=None, hurdle=None)
        reduction["hilo"] = hilo

        # Populate reduction df
        n += 1
        hi_lvl_col = str(_h)[:2] + str(n)
        lo_lvl_col = str(_l)[:2] + str(n)

        reduce_hi = reduction.loc[reduction["hilo"] < 0, _h]
        reduce_lo = reduction.loc[reduction["hilo"] > 0, _l]
        reduction[hi_lvl_col] = reduce_hi
        reduction[lo_lvl_col] = reduce_lo

        # Populate main dataframe
        df[hi_lvl_col] = reduce_hi
        df[lo_lvl_col] = reduce_lo

        # Reduce reduction
        reduction = reduction.dropna(subset=["hilo"])
        reduction.fillna(method="ffill", inplace=True)
        highs = reduction[hi_lvl_col].values
        lows = -reduction[lo_lvl_col].values

        if n >= lvl_limit:
            break

    return df


def cleanup_latest_swing(df, shi, slo, rt_hi, rt_lo):
    """
    removes false positives
    """
    # latest swing
    try:
        shi_dt = df.loc[pd.notnull(df[shi]), shi].index[-1]
        s_hi = df.loc[pd.notnull(df[shi]), shi][-1]
        slo_dt = df.loc[pd.notnull(df[slo]), slo].index[-1]
        s_lo = df.loc[pd.notnull(df[slo]), slo][-1]
    except (IndexError, KeyError):
        raise NotEnoughDataError

    len_shi_dt = len(df[:shi_dt])
    len_slo_dt = len(df[:slo_dt])

    # Reset false positives to np.nan
    for _ in range(2):
        if (
            len_shi_dt > len_slo_dt and
            (
                df.loc[shi_dt:, rt_hi].max() > s_hi or s_hi < s_lo
            )
        ):
            df.loc[shi_dt, shi] = np.nan
            len_shi_dt = 0
        elif (
            len_slo_dt > len_shi_dt and
            (
                df.loc[slo_dt:, rt_lo].min() < s_lo or s_hi < s_lo
            )
        ):
            df.loc[slo_dt, slo] = np.nan
            len_slo_dt = 0
        else:
            pass

    return df


def latest_swing_variables(df, shi, slo, rt_hi, rt_lo, _h='high', _l='low', _c='close'):
    """

    :param df:
    :param shi:
    :param slo:
    :param rt_hi:
    :param rt_lo:
    :param _h:
    :param _l:
    :param _c:
    :return:
    """
    try:
        # get that latest swing hi/lo dates
        shi_query = pd.notnull(df[shi])
        slo_query = pd.notnull(df[slo])

        shi_dt = df.loc[shi_query, shi].index[-1]
        slo_dt = df.loc[slo_query, slo].index[-1]

        s_hi = df.loc[shi_query, shi][-1]
        s_lo = df.loc[slo_query, slo][-1]
    except IndexError:
        raise NotEnoughDataError

    if slo_dt > shi_dt:
        # swing low date is more recent
        swg_var = [
            1,
            s_lo,
            slo_dt,
            rt_lo,
            shi,
            df.loc[slo_dt:, _h].max(),
            df.loc[slo_dt:, _h].idxmax(),
            'high'
        ]
    elif shi_dt > slo_dt:
        swg_var = [
            -1,
            s_hi,
            shi_dt,
            rt_hi,
            slo,
            df.loc[shi_dt:, _l].min(),
            df.loc[shi_dt:, _l].idxmin(),
            'low'
        ]
    else:
        swg_var = [0] * 7

    return swg_var


def retest_swing(
        df,
        ud: int,
        _rt,
        hh_ll_dt: pd.Timestamp,
        hh_ll: float,
        _swg: str,
        idx_extreme_f: str,
        extreme_f: str,
        cum_extreme_f: str,
        _c: str = 'close',
):
    """
    :param ud:
    :param cum_extreme_f:
    :param extreme_f:
    :param idx_extreme_f:
    :param df:
    :param _rt:
    :param hh_ll_dt: date of hh_ll
    :param hh_ll: lowest low or highest high
    :param _c: close col str
    :param _swg: series to assign the value, shi for swing hi, slo for swing lo
    :return:
    """
    rt_sgmt = df.loc[hh_ll_dt:, _rt]
    discovery_lag = None

    if rt_sgmt.count() > 0:  # Retests exist and distance test met
        rt_dt = getattr(rt_sgmt, idx_extreme_f)()
        rt_hurdle = getattr(rt_sgmt, extreme_f)()
        rt_px = getattr(df.loc[rt_dt:, _c], cum_extreme_f)()
        df.loc[rt_dt, "rt"] = rt_hurdle

        breach_query = (np.sign(rt_px - rt_hurdle) == -np.sign(ud))
        discovery_lag = rt_sgmt.loc[breach_query].first_valid_index()

        if discovery_lag is not None:
            df.at[hh_ll_dt, _swg] = hh_ll

    return df, discovery_lag


def retrace_swing(
        df,
        ud,
        _swg,
        hh_ll_dt,
        hh_ll,
        vlty,
        retrace_vol,
        retrace_pct,
        _c='close'
):
    """

    :param df:
    :param ud:
    :param _swg:
    :param hh_ll_dt:
    :param hh_ll:
    :param vlty: volatility at hh_ll_dt
    :param retrace_vol: volatility multiplied by some value
    :param retrace_pct:
    :param _c:
    :return:
    """
    if ud == 1:
        extreme_f = 'min'
        extreme_idx_f = 'idxmin'

        def f(divisor, _):
            return abs(retrace / divisor)
    # else ub assumed to be -1
    else:
        extreme_f = 'max'
        extreme_idx_f = 'idxmax'

        def f(divisor, round_val):
            return round(retrace / divisor, round_val)

    data_range = df.loc[hh_ll_dt:, _c]
    retrace = getattr(data_range, extreme_f)() - hh_ll
    discovery_lag = None

    if (
        vlty > 0 and
        retrace_vol > 0 and
        f(vlty, 1) - retrace_vol > 0 or
        retrace_pct > 0 and
        f(hh_ll, 4) - retrace_pct > 0
    ):
        discovery_lag = getattr(data_range, extreme_idx_f)()
        df.at[hh_ll_dt, _swg] = hh_ll

    # if _sign == 1:  #
    #     retrace = df.loc[hh_ll_dt:, _c].min() - hh_ll
    #     if (
    #             (vlty > 0)
    #             & (retrace_vol > 0)
    #             & ((abs(retrace / vlty) - retrace_vol) > 0)
    #     ):
    #         df.at[hh_ll_dt, _swg] = hh_ll
    #     elif (retrace_pct > 0) & ((abs(retrace / hh_ll) - retrace_pct) > 0):
    #         df.at[hh_ll_dt, _swg] = hh_ll
    #
    # elif _sign == -1:
    #     retrace = df.loc[hh_ll_dt:, _c].max() - hh_ll
    #     if (
    #             (vlty > 0)
    #             & (retrace_vol > 0)
    #             & ((round(retrace / vlty, 1) - retrace_vol) > 0)
    #     ):
    #         df.at[hh_ll_dt, _swg] = hh_ll
    #     elif (retrace_pct > 0) & ((round(retrace / hh_ll, 4) - retrace_pct) > 0):
    #         df.at[hh_ll_dt, _swg] = hh_ll
    # else:
    #     retrace = 0
    return df, discovery_lag


def new_retrace_swing(close_price, swing_params, prev_swing_data, retrace_vlty, dist_vlty, dist_pct, retrace_pct):
    """
    continues calculation of retracement
    dfs should be restarted to most recent swing
    """
    distance = swing_params.adj_sub(prev_swing_data.st_px)
    dist_vlty_test = (distance - dist_vlty) > 0
    pct_test = ((distance - 1) - dist_pct) > 0
    retrace = swing_params.adj_sub(close_price)

    # TODO atr levels should be base vlty
    vlty_breach = (retrace / swing_params.base_atr_levels) - retrace_vlty
    vlty_breach = vlty_breach.loc[dist_vlty_test & (vlty_breach > 0)]
    vlty_breach_date = vlty_breach.first_valid_index()

    pct_breach = (retrace / swing_params.extreme) - retrace_pct
    pct_breach = pct_breach.loc[pct_test & (pct_breach > 0)]
    pct_breach_date = pct_breach.first_valid_index()
    discovery_date = min(normalize_none_values([vlty_breach_date, pct_breach_date]))
    return discovery_date


def new_retest_swing(close_price, swing_params, prev_swing_price, dist_vlty):
    """retest table required"""
    distance = swing_params.adj_sub(prev_swing_price)
    dist_vlty_test = (distance - dist_vlty) > 0
    if not swing_params.rt.empty:
        cum_hurdle = getattr(swing_params.rt, 'cummax' if swing_params.type == -1 else 'cummin')()
        breach_query = swing_params.adj(close_price - cum_hurdle)
        discovery_date = swing_params


def normalize_none_values(values):
    """
    when comparing a set of indexes received from first_valid_date(),
    normalize any None values to a valid value to avoid error
    """
    not_none_val = None
    for val in values:
        if val is not None:
            not_none_val = val
            break
    res = None
    if not_none_val is not None:
        res = [val if val is not None else not_none_val for val in values]
    return res


def test_distance(base_sw_val, hh_ll, dist_vol, dist_pct):
    """
    when swing low is latest, does the highest high afterward exceed the distance test?
    :param ud: direction
    :param base_sw_val: base, swing hi/lo
    :param hh_ll: lowest low or highest high
    :param dist_vol:
    :param dist_pct:
    :return:
    """
    # priority: 1. Vol 2. pct 3. dflt
    if dist_vol > 0:
        distance_test = np.sign(abs(hh_ll - base_sw_val) - dist_vol)
    elif dist_pct > 0:
        distance_test = np.sign(abs(hh_ll / base_sw_val - 1) - dist_pct)
    else:
        distance_test = np.sign(dist_pct)
    return distance_test > 0


class LatestSwingData:
    def __init__(
            self,
            ud,  # direction +1up, -1down
            base_sw,  # base, swing hi/lo
            bs_dt,  # swing date
            _rt,  # series name used to detect swing, rt_lo for swing hi, rt_hi for swing lo
            sw_col,  # series to assign the value, shi for swing hi, slo for swing lo
            hh_ll,  # lowest low or highest high
            hh_ll_dt,  # date of hh_ll
            price_col,
    ):
        self.ud = ud

        if self.ud == 1:
            extreme = 'max'
            cum_extreme = 'min'
        else:
            extreme = 'min'
            cum_extreme = 'max'

        self.idx_extreme_f = f'idx{extreme}'
        self.extreme_f = f'{extreme}'
        self.cum_extreme_f = f'cum{cum_extreme}'

        self.base_sw = base_sw
        self.bs_dt = bs_dt
        self.rt = _rt
        self.sw_col = sw_col
        self.extreme_val = hh_ll
        self.extreme_date = hh_ll_dt
        self.price_col = price_col

    @classmethod
    def init_from_latest_swing(cls, df, shi='hi3', slo='lo3', rt_hi='hi1', rt_lo='lo1', _h='high', _l='low', _c='close'):
        shi_query = pd.notnull(df[shi])
        slo_query = pd.notnull(df[slo])
        try:
            # get that latest swing hi/lo dates
            sw_hi_date = df.loc[shi_query, shi].index[-1]
            sw_lo_date = df.loc[slo_query, slo].index[-1]
        except IndexError:
            raise NotEnoughDataError

        if sw_lo_date > sw_hi_date:
            # swing low date is more recent
            s_lo = df.loc[slo_query, slo][-1]
            swg_var = cls(
                ud=1,
                base_sw=s_lo,
                bs_dt=sw_lo_date,
                _rt=rt_lo,
                sw_col=shi,
                hh_ll=df.loc[sw_lo_date:, _h].max(),
                hh_ll_dt=df.loc[sw_lo_date:, _h].idxmax(),
                price_col='high'
            )
        #  (shi_dt > slo_dt) assuming that shi_dt == slo_dt is impossible
        else:
            s_hi = df.loc[shi_query, shi][-1]
            swg_var = cls(
                ud=-1,
                base_sw=s_hi,
                bs_dt=sw_hi_date,
                _rt=rt_hi,
                sw_col=slo,
                hh_ll=df.loc[sw_hi_date:, _l].min(),
                hh_ll_dt=df.loc[sw_hi_date:, _l].idxmin(),
                price_col='low'
            )
        return swg_var

    def test_distance(self, dist_vol, dist_pct):
        return test_distance(
            base_sw_val=self.base_sw,
            hh_ll=self.extreme_val,
            dist_vol=dist_vol,
            dist_pct=dist_pct
        )

    def retest_swing(self, df):
        return retest_swing(
            df=df,
            ud=self.ud,
            _rt=self.rt,
            hh_ll_dt=self.extreme_date,
            hh_ll=self.extreme_val,
            _swg=self.sw_col,
            idx_extreme_f=self.idx_extreme_f,
            extreme_f=self.extreme_f,
            cum_extreme_f=self.cum_extreme_f
        )

    def retrace_swing(self, df, vlty, retrace_vol, retrace_pct):
        return retrace_swing(
            df=df,
            ud=self.ud,
            _swg=self.sw_col,
            hh_ll_dt=self.extreme_date,
            hh_ll=self.extreme_val,
            vlty=vlty,
            retrace_vol=retrace_vol,
            retrace_pct=retrace_pct,
        )

    def volatility_swing(self, df, dist_pct, vlty, retrace_pct, retrace_vol_mult=2.5, dist_vol_mult=5):
        """detect last swing via volatility test"""
        dist_vol = vlty * dist_vol_mult
        res = self.test_distance(dist_vol, dist_pct)
        if res is True:
            retrace_vol = vlty * retrace_vol_mult
            df = self.retest_swing(df)
            df = self.retrace_swing(df, vlty=vlty, retrace_vol=retrace_vol, retrace_pct=retrace_pct)
        return df


def old_init_swings(
        df: pd.DataFrame,
        dist_pct: float,
        retrace_pct: float,
        n_num: int,
        lvl=3,
        lvl_limit=3,
):
    _o, _h, _l, _c = ['open', 'high', 'low', 'close']
    shi = f'hi{lvl}'
    slo = f'lo{lvl}'
    rt_hi = 'hi1'
    rt_lo = 'lo1'

    df = historical_swings(df, lvl_limit=lvl_limit)
    df = cleanup_latest_swing(df, shi=shi, slo=slo, rt_hi=rt_hi, rt_lo=rt_lo)
    # latest_sw_vars = LatestSwingData.init_from_latest_swing(df, shi, slo, rt_hi, rt_lo)
    # volatility_series = average_true_range(df=df, window=n_num)
    # _dist_vol_series = volatility_series * 5
    # df['rol_hi'] = df['high'].rolling(n_num).max()
    # df['rol_lo'] = df['low'].rolling(n_num).min()
    #
    # df['hi_vol'] = (df['rol_hi'] - _dist_vol_series).ffill()
    # df['lo_vol'] = (df['rol_lo'] + _dist_vol_series).ffill()
    # _retrace_vol_series = volatility_series * 2.5
    # vlty = round(volatility_series[latest_sw_vars.extreme_date], 2)

    # px = df.loc[bs_dt: hh_ll_dt, price_col]
    # vol = _dist_vol_series.loc[bs_dt: hh_ll_dt]
    # _df = pd.DataFrame()
    # diff = np.sign(abs(px - base_sw) - vol)
    # _df['diff'] = diff
    # _df.diff = _df.loc[(_df.diff > 0)]
    # diff = np.where(diff > 0, 1, 0) * ud
    # vol_lvl = base_sw - vol
    # _t = pd.DataFrame({
    #     price_col: px,
    #     'vlty_test': diff,
    #     'vol_lvl': vol_lvl
    # })
    # _t['base'] = base_sw
    # discovery_lag = None
    # dist_vol = vlty * 5
    # res = latest_sw_vars.test_distance(dist_vol, dist_pct)
    # if res is True:
    #     retrace_vol = vlty * 2.5
    #     df, retest_swing_lag = latest_sw_vars.retest_swing(df)
    #     df, retrace_swing_lag = latest_sw_vars.retrace_swing(
    #         df, vlty=vlty, retrace_vol=retrace_vol, retrace_pct=retrace_pct
    #     )
    #     lag_compare = []
    #     if retest_swing_lag is not None:
    #         lag_compare.append(retest_swing_lag)
    #
    #     if retrace_swing_lag is not None:
    #         lag_compare.append(retrace_swing_lag)
    #
    #     if len(lag_compare) > 0:
    #         discovery_lag = np.maximum(*lag_compare)

    return df, discovery_lag
