"""
contains original code by Laurent Bernut relating to
swing and regime definition
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import typing

import typing as t
import pandas_accessors.accessors as pda
from abc import ABC, abstractmethod
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
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    """
    _max = df[_h].combine(df[_c].shift(), max)
    _min = df[_l].combine(df[_c].shift(), min)
    atr = (_max - _min).rolling(window=window).mean()
    return atr


def relative(
        df: pd.DataFrame,
        bm_df: pd.DataFrame,
        bm_col: str = "close",
        _o: str = "open",
        _h: str = "high",
        _l: str = "low",
        _c: str = "close",
        ccy_df: typing.Optional[pd.DataFrame] = None,
        ccy_col: typing.Optional[str] = None,
        dgt: typing.Optional[int] = None,
        rebase: typing.Optional[bool] = True,
) -> pd.DataFrame:
    """
    df: df
    bm_df, bm_col: df benchmark dataframe & column name
    ccy_df,ccy_col: currency dataframe & column name
    dgt: rounding decimal
    start/end: string or offset
    rebase: boolean rebase to beginning or continuous series
    """

    # BJ: No, input dataframe should already be sliced
    # # Slice df dataframe from start to end period: either offset or datetime
    # df = df[start:end]

    # inner join of benchmark & currency: only common values are preserved
    df = df.join(bm_df[[bm_col]], how="inner")
    adjustment = df[bm_col].copy()
    if ccy_df is not None:
        df = df.join(ccy_df[[ccy_col]], how="inner")
        adjustment = df[bm_col].mul(df["ccy"])
        if dgt is not None:
            adjustment = round(adjustment, dgt)

    # rename benchmark name as bm and currency as ccy
    # df.rename(columns={bm_col: bm_col, ccy_col: "ccy"}, inplace=True)

    # Adjustment factor: calculate the scalar product of benchmark and currency

    df["bmfx"] = adjustment.fillna(method="ffill")

    if rebase is True:
        df["bmfx"] = df["bmfx"].div(df["bmfx"].iloc[0])

    # Divide absolute price by fxcy adjustment factor and rebase to first value
    _ro = "r" + str(_o)
    _rh = "r" + str(_h)
    _rl = "r" + str(_l)
    _rc = "r" + str(_c)

    df[_ro] = df[_o].div(df["bmfx"])
    df[_rh] = df[_h].div(df["bmfx"])
    df[_rl] = df[_l].div(df["bmfx"])
    df[_rc] = df[_c].div(df["bmfx"])

    if dgt is not None:
        df["r" + str(_o)] = round(df["r" + str(_o)], dgt)
        df["r" + str(_h)] = round(df["r" + str(_h)], dgt)
        df["r" + str(_l)] = round(df["r" + str(_l)], dgt)
        df["r" + str(_c)] = round(df["r" + str(_c)], dgt)

    # drop after function is called, user decides
    # df = df.drop([bm_col, "ccy", "bmfx"], axis=1)

    return df


def simple_relative(df, bm_close, rebase=True):
    """simplified version of relative calculation"""
    bm = bm_close.ffill()
    if rebase is True:
        bm = bm.div(bm[0])
    return df.div(bm, axis=0)


def relative_all_rebase(df, bm_close, axis):
    return df.div(bm_close * df.iloc[0], axis=axis) * bm_close[0]


class RetraceSwingCalc:
    def __init__(self, dist_pct, retrace_pct, retrace_vlty_mult, dist_vlty_mult):
        self._retrace_vlty_multiplier = retrace_vlty_mult
        self._dist_vlty_mult = dist_vlty_mult
        self._dist_pct = dist_pct
        self._retrace_vlty = retrace_pct


class AbcSwingParams:
    def __init__(self, extreme_levels, atr_levels,
                 peaks, sw_type, dist_vlty_mult, retrace_vlty_mult, dist_pct, retrace_pct
                 ):
        self.extreme_levels = extreme_levels
        self.base_atr_levels = atr_levels
        self.dist_atr_levels = atr_levels * dist_vlty_mult
        self.retrace_atr_levels = atr_levels * retrace_vlty_mult
        self.dist_pct = dist_pct
        self.retrace_pct = retrace_pct
        self.peaks = peaks
        self.sw_type = sw_type
        self.dist_vlty_mult = dist_vlty_mult
        self.retrace_vlty_mult = retrace_vlty_mult

    @abstractmethod
    def update_params(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_cum_f(sw_type):
        assert sw_type in [-1, 1]
        if sw_type == 1:
            res = {'cum_f': 'cummin', 'retest_cum_f': 'cummax'}
        else:
            res = {'cum_f': 'cummax', 'retest_cum_f': 'cummin'}
        return res

    def adj(self, val):
        return val * self.sw_type

    def adj_sub(self, price):
        """diff values adjusted by swing type for symmetrical sw_hi/sw_lo calculations"""
        return self.adj(price - self.extreme_levels)

    def get_peak_date(self, peak_discovery_date):
        """given a date, returns the first date of the cummax/cummin this date falls under"""
        peak_date = None
        if peak_discovery_date is not None:
            date_query = self.peaks.index <= peak_discovery_date
            price_query = self.peaks == self.extreme_levels.loc[peak_discovery_date]
            peak_date = self.peaks.loc[date_query & price_query].index[-1]

        # when average price is used...
        # date_query = px.index <= peak_discovery_date
        # price_query = px.avg_px == extremes.loc[peak_discovery_date]
        # peak_date = px.loc[date_query & price_query].iloc[-1].name
        return peak_date

    def get_next_peak_data(self, close_price, prev_swing_price):
        retrace = self.adj_sub(close_price)
        distance_threshold = retrace - self.dist_atr_levels
        pct_threshold = (retrace / self.extreme_levels) - self.retrace_pct
        peak_discovery = close_price.loc[
            (distance_threshold > 0) |
            (pct_threshold > 0)
        ]
        peak_discovery_date = peak_discovery.first_valid_index()
        peak_date = self.get_peak_date(peak_discovery_date)
        dist_breach = None
        pct_breach = None
        if peak_discovery_date is not None:
            dist_breach = distance_threshold.at[peak_discovery_date]
            pct_breach = pct_threshold.at[peak_discovery_date]
        return {'peak_date': peak_date,
                'peak_discovery_date': peak_discovery_date,
                'thresh_breach': dist_breach,
                'pct_breach': pct_breach}

    def get_next_retrace_swing(self, close_price, prev_swing_price):
        distance = self.adj_sub(prev_swing_price)
        dist_vlty_test = (distance - self.dist_atr_levels) > 0
        pct_test = ((distance - 1) - self.dist_pct) > 0
        retrace = self.adj_sub(close_price)

        # TODO atr levels should be base vlty
        vlty_breach = (retrace / self.base_atr_levels) - self.retrace_atr_levels
        vlty_breach = vlty_breach.loc[dist_vlty_test & (vlty_breach > 0)]
        vlty_breach_date = vlty_breach.first_valid_index()

        pct_breach = (retrace / self.extreme_levels) - self.retrace_pct
        pct_breach = pct_breach.loc[pct_test & (pct_breach > 0)]
        pct_breach_date = pct_breach.first_valid_index()
        peak_discovery_date = normalize_none_values([vlty_breach_date, pct_breach_date])
        if peak_discovery_date is not None:
            peak_discovery_date = min(peak_discovery_date)
        peak_date = self.get_peak_date(peak_discovery_date)
        return {'peak_date': peak_date, 'peak_discovery_date': peak_discovery_date}


class BaseSwingParams(AbcSwingParams):
    def __init__(self, atr: pd.Series, price: pd.Series, sw_type: int, dist_vlty_mult, retrace_vlty_mult, dist_pct, retrace_pct):
        cum_funcs = self.__class__.get_cum_f(sw_type)
        self._retest_cum_f = cum_funcs['retest_cum_f']
        param_attrs = self.__class__._init_params(atr, price, sw_type)
        _atr_valid_dt = atr.first_valid_index()
        _px = price.loc[_atr_valid_dt:]
        _atr = atr.loc[_atr_valid_dt:]
        param_attrs['extreme_levels'] = param_attrs['extreme_levels'].loc[_atr_valid_dt:]
        param_attrs['atr_levels'] = param_attrs['atr_levels'].loc[_atr_valid_dt:]
        param_attrs['sw_type'] = sw_type
        super().__init__(dist_vlty_mult=dist_vlty_mult, retrace_vlty_mult=retrace_vlty_mult,
                         dist_pct=dist_pct, retrace_pct=retrace_pct, **param_attrs)
        self._base_atr = _atr
        self._base_price = _px

    @staticmethod
    def _init_params(atr: pd.Series, price: pd.Series, sw_type: int):
        extremes = (price * sw_type).cummin() * sw_type
        extremes_changed = extremes != extremes.shift(1)
        peaks = price.loc[extremes_changed]
        _atr_at_peaks = atr.copy()
        _atr_at_peaks.loc[~extremes_changed] = np.nan
        _atr_at_peaks = _atr_at_peaks.ffill()
        return {
            'extreme_levels': extremes,
            'atr_levels': _atr_at_peaks,
            'peaks': peaks,
        }

    def update_params(self, date_index):
        self._base_price = self._base_price.loc[self._base_price.index > date_index]
        self._base_atr = self._base_atr.loc[self._base_atr.index > date_index]
        res = self.__class__._init_params(self._base_atr, self._base_price, self.sw_type)
        self.extreme_levels = res['extreme_levels']
        self.base_atr_levels = res['atr_levels']
        self.dist_atr_levels = self.base_atr_levels * self.dist_vlty_mult
        self.retrace_vlty_mult = self.base_atr_levels * self.retrace_vlty_mult
        self.peaks = res['peaks']


class DerivedSwingParams(BaseSwingParams):
    def __init__(self, retest_peaks, **kwargs):
        # self._cum_f = self.__class__.get_cum_f(sw_type)
        # param_attrs = self.__class__._init_params(index, atr, raw_peaks, self._cum_f)
        # param_attrs['sw_type'] = sw_type
        # super().__init__(**param_attrs)
        # self._base_atr = atr
        # self._base_index = index
        super().__init__(**kwargs)
        self.retest = retest_peaks

    # @staticmethod
    # def _init_params(index, atr, raw_peaks, cum_f):
    #     """
    #     :param index: index of price series
    #     :param atr: average true range series
    #     :param raw_peaks: peaks discovered on the lower level
    #     :return:
    #     """
    #     extreme_levels = pd.DataFrame(index=index, columns=['ext'])
    #     extreme_levels['ext'] = raw_peaks
    #     extreme_levels = extreme_levels['ext']
    #     extreme_levels = getattr(extreme_levels, cum_f)().ffill()
    #     atr_levels = atr.copy()
    #     atr_levels.loc[~atr_levels.index.isin(raw_peaks.index)] = np.nan
    #     atr_levels = atr_levels.ffill()
    #     return {'extreme_levels': extreme_levels, 'atr_levels': atr_levels, 'peaks': raw_peaks}

    # def update_params(self, date_index):
    #     self._base_index = self._base_index[self._base_index.get_loc(date_index) + 1:]
    #     self.peaks = self.peaks.loc[self.peaks.index > date_index]
    #     res = self.__class__._init_params(self._base_index, self._base_atr, self.peaks, self._cum_f)
    #     self.extreme_levels = res['extreme_levels']
    #     self.base_atr_levels = self.base_atr_levels[self.base_atr_levels.index > date_index]

    def update_params(self, date_index):
        super().update_params(date_index)
        self.retest = self.retest.loc[self.retest.index > date_index]

    @staticmethod
    def swing_to_raw_peaks(swing_table) -> t.Tuple[pd.Series, pd.Series]:
        raw_peaks = swing_table.copy()
        raw_peaks.index = raw_peaks.start
        sw_hi_peaks = raw_peaks.loc[raw_peaks.type == -1]
        sw_lo_peaks = raw_peaks.loc[raw_peaks.type == 1]
        return sw_hi_peaks.st_px, sw_lo_peaks.st_px

    # def get_next_peak_data(self, close_price, prev_swing_price):
    #     """retest table required"""
    #     distance = self.adj_sub(prev_swing_price)
    #     dist_vlty_test = (distance - self.dist_atr_levels) > 0
    #     res = {'peak_date': None, 'peak_discovery_date': None}
    #     if not self.retest.empty:
    #         # cum_hurdle = getattr(self.retest, 'cummax' if self.sw_type == -1 else 'cummin')()
    #         cum_hurdle = self.retest.ffill()
    #         breach_query = self.adj(close_price - cum_hurdle)
    #         peak_discovery_date = close_price.loc[dist_vlty_test & (breach_query > 0)].first_valid_index()
    #         peak_date = self.get_peak_date(peak_discovery_date)
    #         res = {'peak_date': peak_date, 'peak_discovery_date': peak_discovery_date}
    #     return res


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

    def retest(self):
        """apply retest stuff"""
        pass


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


def historical_swings(df, _h='high', _l='low', _c='close', lvl_limit=3):
    reduction = df[list({_h, _l, _c})].copy()
    highs = reduction[_h]
    lows = reduction[_l]
    reduction_target = len(reduction) // 100

    def init_compare(swings, lvl):
        compare = pd.DataFrame(columns=['prior', 'sw', 'next'])
        compare.next = swings.shift(-1)
        if lvl == 1:
            swings = swings.loc[swings.shift(-1) != swings]
        compare.prior = swings.shift(1)
        compare.sw = swings
        return compare.dropna()

    n = 0
    while len(reduction) >= reduction_target:
        n += 1
        hi_lvl_col = 'hi' + str(n)
        lo_lvl_col = 'lo' + str(n)

        high_compare = init_compare(highs, 1)
        highs = high_compare.loc[
            (high_compare.sw > high_compare.prior) &
            (high_compare.sw > high_compare.next),
            'sw'
        ]
        low_compare = init_compare(lows, 1)
        lows = low_compare.loc[
            (low_compare.sw < low_compare.prior) &
            (low_compare.sw < low_compare.next),
            'sw'
        ]

        # Populate main dataframe
        df[hi_lvl_col] = highs
        df[lo_lvl_col] = lows

        if n >= lvl_limit:
            break

    return df


def full_peak_lag(df, asc_peaks) -> pd.DataFrame:
    """
    calculates distance from highest level peak to the time it was discovered
    value is not nan if peak, does not matter if swing low or swing high
    :param df:
    :param asc_peaks: peak level columns in ascending order
    :return:
    """
    # desire lag for all peak levels greater than 1,
    # so if [hi1, hi2, hi3] given,
    # group by [[hi1, hi2], [hi1, hi2, hi3]] to get lag for level 2 and level 3
    lowest_lvl = asc_peaks[0]
    start_peaks = df.loc[df[lowest_lvl].notna(), lowest_lvl]
    sw_type = 1 if 'lo' in lowest_lvl else -1
    lvl = 1
    lowest_lvl_peaks = pd.DataFrame(
        data={
            'start': start_peaks.index,
            'end': start_peaks.index + 1,
            'type': sw_type,
            'lvl': lvl,
            'st_px': start_peaks.values,
            'en_px': df.loc[start_peaks.index + 1, 'close'].values
        }
    )

    peak_tables = [lowest_lvl_peaks]
    lower_level_peaks = lowest_lvl_peaks[['start', 'end']].copy()
    for peak_col in asc_peaks[1:]:
        lvl += 1
        lower_level_peaks.end = lower_level_peaks.end.shift(-1)
        try:
            peaks = df.loc[df[peak_col].notna()]
        except KeyError:
            raise NotEnoughDataError
        peak_table = pd.DataFrame(
            data={
                'start': peaks.index,
                'type': sw_type,
                'lvl': lvl,
                'st_px': df.loc[peaks.index, 'close'],
            }
        )
        peak_table = peak_table.merge(right=lower_level_peaks, how='left', on=['start'])
        peak_table.end = peak_table.end.astype(dtype='int')
        peak_table['en_px'] = df.loc[peak_table.end, 'close'].values
        peak_tables.append(peak_table)
        lower_level_peaks = peak_tables[-1][['start', 'end']].copy()

    full_pivot_table = pd.concat(peak_tables)
    return full_pivot_table


def get_follow_peaks(
    current_peak: pd.Series, prior_peaks: pd.Series
) -> t.Tuple[pd.Series, pd.DataFrame]:
    """
    calculates lage between current peak and next level peak.
    helper function, must be used sequentially from current level down to lvl 1 peak
    to get full lag
    :param current_peak:
    :param prior_peaks:
    :return:
    """
    pivot_table = pd.DataFrame(columns=[current_peak.name, prior_peaks.name])
    follow_peaks = pd.Series(index=current_peak.index, dtype=pd.Float64Dtype())

    for r in current_peak.dropna().iteritems():
        # slice df starting with r swing, then exclude r swing, drop nans, then only keep the first row
        # gets the first swing of the prior level after the current swing of the current level.
        current_peak_date = r[0]
        follow_peak = prior_peaks.loc[current_peak_date:].iloc[1:].dropna().iloc[:1]
        if len(follow_peak) > 0:
            follow_peaks.loc[follow_peak.index[0]] = follow_peak.iloc[0]
            df = pd.DataFrame({
                    prior_peaks.name: [follow_peak.index[0]],
                    current_peak.name: [current_peak_date],
            })
            pivot_table = pd.concat([pivot_table, df], axis=0, ignore_index=True)
    return follow_peaks, pivot_table


def init_swings(
        df,
        dist_pct,
        retrace_pct,
        n_num,
        lvl=3,
        lvl_limit=3
):
    """
    new init swings function
    :param df:
    :param dist_pct:
    :param retrace_pct:
    :param n_num:
    :param lvl:
    :param lvl_limit:
    :return:
    """
    px = df[['close', 'high', 'low']].copy()
    px['avg_px'] = df[['high', 'low', 'close']].mean(axis=1).reset_index(drop=True)

    lvl_limit = 4
    hi_cols = [f'hi{i}' for i in range(1, lvl_limit + 1)]
    lo_cols = [f'lo{i}' for i in range(1, lvl_limit + 1)]

    lvl1_peaks = historical_swings(px, lvl_limit=4, _h='close', _l='close').reset_index(drop=True)
    if len(lvl1_peaks.hi3.dropna()) == 0 or len(lvl1_peaks.lo3.dropna()) == 0:
        raise NotEnoughDataError

    hi_peaks = full_peak_lag(lvl1_peaks, hi_cols)
    lo_peaks = full_peak_lag(lvl1_peaks, lo_cols)
    peak_table = pd.concat([hi_peaks, lo_peaks]).sort_values(by='end', ascending=True).reset_index(drop=True)

    return px, peak_table


def volatility_swings(base_px, px: pd.DataFrame, hi_sw_params: AbcSwingParams, lo_sw_params, initial_price):
    """
    alternate looking for swing hi/lo starting with whichever swing is found sooner
    :param px:
    :param hi_sw_params:
    :param lo_sw_params:
    :return:
    """
    # TODO need separate param when base vlty is needed for retrace
    # atr = atr * vol_mult
    _px = px.copy()
    # _atr = atr.copy()
    swing_data = []
    latest_swing_data, swing_params = initial_volatility_swing(_px, hi_sw_params, lo_sw_params, initial_price)
    while None not in latest_swing_data.values():
        latest_swing_data['type'] = swing_params.sw_type
        swing_data.append(latest_swing_data.values())
        swing_params = hi_sw_params if swing_params == lo_sw_params else lo_sw_params
        latest_swing_date = latest_swing_data['peak_date']
        latest_swing_discovery_date = latest_swing_data['peak_discovery_date']
        # try:
        last_sw_price = base_px.close[latest_swing_date]
        _px = _px.loc[_px.index > latest_swing_date]
        swing_params.update_params(latest_swing_date)
        # except TypeError:
        #     pass
        # swap swing type from previous and search

        # latest_swing_data = get_next_peak_data(_px.close, swing_params)
        latest_swing_data = swing_params.get_next_peak_data(_px.close, last_sw_price)

        if None in latest_swing_data.values():
            break
        elif latest_swing_data['peak_discovery_date'] < latest_swing_discovery_date:
            latest_swing_data['peak_discovery_date'] = latest_swing_discovery_date

    return pd.DataFrame(data=swing_data, columns=['start', 'end', 'vlty_break', 'pct_break', 'type'])


def initial_volatility_swing(_px, hi_sw_params, lo_sw_params, initial_price):
    """get data for the first swing in the series"""
    # high_peak_data = get_next_peak_data(_px.close, hi_sw_params)
    # low_peak_data = get_next_peak_data(_px.close, lo_sw_params)
    high_peak_data = hi_sw_params.get_next_peak_data(_px.close, initial_price)
    low_peak_data = lo_sw_params.get_next_peak_data(_px.close, initial_price)
    swing_data_selector = {
        high_peak_data['peak_discovery_date']: (high_peak_data, hi_sw_params),
        low_peak_data['peak_discovery_date']: (low_peak_data, lo_sw_params),
    }
    discovery_compare = []
    if None not in high_peak_data.values():
        discovery_compare.append(high_peak_data['peak_discovery_date'])
    if None not in low_peak_data.values():
        discovery_compare.append(low_peak_data['peak_discovery_date'])
    if len(discovery_compare) > 0:
        if len(discovery_compare) > 1:
            latest_swing_discovery_date = np.minimum(*discovery_compare)
        else:
            latest_swing_discovery_date = discovery_compare[0]
        res = swing_data_selector[latest_swing_discovery_date]
    else:
        # if none in both, just return one of None data dicts
        res = swing_data_selector[high_peak_data['peak_discovery_date']]
    return res


def get_next_peak_data(close_price, swing_params: AbcSwingParams) -> t.Dict[str, t.Any]:
    """
    returns the first date where close price crosses distance threshold
    """
    distance_threshold = abs(close_price - swing_params.extreme_levels) - swing_params.base_atr_levels
    peak_discovery_date = close_price.loc[
        (distance_threshold > 0)
    ].first_valid_index()
    if peak_discovery_date is not None:
        peak_date = swing_params.get_peak_date(peak_discovery_date)
    else:
        return {'peak_date': None, 'peak_discovery_date': None}

    return {'peak_date': peak_date, 'peak_discovery_date': peak_discovery_date}


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


@dataclass
class RegimeFcLists:
    fc_vals: t.List
    fc_dates: t.List
    rg_ch_dates: t.List
    rg_ch_vals: t.List

    def update(self, data: t.Dict):
        self.fc_vals.append(data['fc_val'])
        self.fc_dates.append(data['fc_date'])
        self.rg_ch_dates.append(data['rg_ch_date'])
        self.rg_ch_vals.append(data['rg_ch_val'])


def plot(_d, _plot_window=0, _use_index=False, axis=None):
    """"""
    cols = [
        'close',
        "hi3",
        "lo3",
        # "clg",
        # "flr",
        # "rg_ch",
        "rg",
        # 'hi2_lag',
        # 'hi3_lag',
        # 'lo2_lag',
        # 'lo3_lag'
    ]
    _axis = (
        _d[cols]
            .iloc[_plot_window:]
            .plot(
                style=["grey", "ro", "go"],  # "kv", "k^", "c:"],
                figsize=(15, 5),
                secondary_y=["rg"],
                # grid=True,
                # title=str.upper(_ticker),
                use_index=_use_index,
                ax=axis
            )
    )
    return _axis


def regime_floor_ceiling(
        df: pd.DataFrame,
        flr,
        clg,
        rg,
        rg_ch,
        stdev,
        threshold,
        peak_table,
        sw_lvl: int = 3,
        _h: str = "high",
        _l: str = "low",
        _c: str = "close",
):

    _peak_table = peak_table.loc[peak_table.lvl == sw_lvl].sort_values('start')
    # retest_table = peak_table.loc[peak_table.lvl == 1]
    # retest_lo_table = retest_table.loc[retest_table.type == 1]
    # retest_hi_table = retest_table.loc[retest_table.type == -1]
    _sw_hi_peak_table = _peak_table.loc[_peak_table.type == -1]
    _sw_lo_peak_table = _peak_table.loc[_peak_table.type == 1]

    _hi_lo_table = pd.DataFrame(
        data={'hi': _sw_hi_peak_table.start, 'lo': _sw_lo_peak_table.start}
    ).ffill().bfill().reset_index(drop=True)


    fc_find_floor = hof_find_fc(
        df=df,
        price_col='close',
        extreme_func='min',
        stdev=stdev,
        sw_type=1,
        threshold=threshold
    )
    fc_find_ceiling = hof_find_fc(
        df=df,
        price_col='close',
        extreme_func='max',
        stdev=stdev,
        sw_type=-1,
        threshold=threshold
    )
    fc_floor_found = hof_fc_found(
        df=df,
        cum_func='cummin',
        fc_type=-1,
        retest='hi1'
    )
    fc_ceiling_found = hof_fc_found(
        df=df,
        cum_func='cummax',
        fc_type=1,
        retest='lo1'
    )
    calc_breakdown = hof_break_pullback(
        df=df,
        extreme_idx_f='idxmin',
        extreme_val_f='cummin',
        retest='hi1'
    )
    calc_breakout = hof_break_pullback(
        df=df,
        extreme_idx_f='idxmax',
        extreme_val_f='cummax',
        retest='lo1'
    )

    # Range initialisation to 1st swing
    fc_data_cols = ['test', 'fc_val', 'fc_date', 'rg_ch_date', 'rg_ch_val', 'type']
    init_floor_data = {col: (df.index[0] if col == 'fc_date' else 0) for col in fc_data_cols}
    init_ceiling_data = init_floor_data.copy()
    init_floor_data['type'] = 1
    init_ceiling_data['type'] = -1

    fc_data = pd.DataFrame(columns=fc_data_cols)
    fc_data = pd.concat(
        [
            fc_data,
            pd.DataFrame(data=init_floor_data, index=[0]),
            pd.DataFrame(data=init_ceiling_data, index=[1]),
        ]
    )

    # Boolean variables
    ceiling_found = floor_found = breakdown = breakout = False

    latest_swing_data = None

    # Loop through swings
    for i in range(len(_hi_lo_table)):
        # asymmetric swing list: default to last swing if shorter list
        sw_lo_data = _sw_lo_peak_table.loc[_hi_lo_table.iat[i, 1] == _sw_lo_peak_table.start]
        sw_lo_data = sw_lo_data.iloc[-1]
        sw_hi_data = _sw_hi_peak_table.loc[_hi_lo_table.iat[i, 0] == _sw_hi_peak_table.start]
        sw_hi_data = sw_hi_data.iloc[-1]

        swing_discovery_date = np.maximum(sw_lo_data.start, sw_hi_data.start)  # latest swing index
        latest_swing_data = sw_lo_data if swing_discovery_date == sw_lo_data.start else sw_hi_data

        # CLASSIC CEILING DISCOVERY
        if ceiling_found is False:
            # Classic ceiling test
            current_floor = fc_data.loc[
                (fc_data.type == 1) &
                (fc_data.fc_date <= sw_hi_data.start)
            ].iloc[-1]
            res = fc_find_ceiling(fc_ix=current_floor.fc_date, latest_swing=sw_hi_data)
            if len(res) > 0:
                # Boolean flags reset
                ceiling_found = True
                floor_found = breakdown = breakout = False
                # Append lists
                fc_data = pd.concat([fc_data, pd.DataFrame(data=res, index=[len(fc_data)])])

                # EXCEPTION HANDLING: price penetrates discovery swing
        # 1. if ceiling found, calculate regime since rg_ch_ix using close.cummin
        elif ceiling_found is True:
            try:
                res, df = fc_ceiling_found(
                    rg_ch_data=fc_data.iloc[-1],
                    latest_hi_lo_sw_discovery=latest_swing_data.end
                )
            except ValueError:
                res = False
            if res is True:
                # 3rd fc_data is the first discovered regime,
                # since rg_ch_val is not necessarily valid since we do not have prior data
                breakout = True
                floor_found = ceiling_found = breakdown = False

        # 3. if breakout, test for bearish pullback from highest high since rg_ch_ix
        if breakout is True:
            df = calc_breakout(
                rg_ch_data=fc_data.iloc[-1],
                latest_sw_discovery=latest_swing_data.end
            )

        # CLASSIC FLOOR DISCOVERY
        if floor_found is False:
            # Classic floor test
            current_ceiling = fc_data.loc[
                (fc_data.type == -1) &
                (fc_data.fc_date <= sw_lo_data.start)
            ].iloc[-1]
            res = fc_find_floor(fc_ix=current_ceiling.fc_date, latest_swing=sw_lo_data)
            if len(res) > 0:
                # Boolean flags reset
                floor_found = True
                ceiling_found = breakdown = breakout = False
                fc_data = pd.concat([fc_data, pd.DataFrame(data=res, index=[len(fc_data)])])

        # EXCEPTION HANDLING: price penetrates discovery swing
        # 1. if floor found, calculate regime since rg_ch_ix using close.cummin
        elif floor_found is True:
            try:
                res, df = fc_floor_found(
                    rg_ch_data=fc_data.iloc[-1],
                    latest_hi_lo_sw_discovery=latest_swing_data.end
                )
            except ValueError:
                res = False
            if res is True:
                # 3rd fc_data is the first discovered regime,
                # since rg_ch_val is not necessarily valid since we do not have prior data
                breakdown = True
                ceiling_found = floor_found = breakout = False

        # 3. if breakdown,test for bullish rebound from lowest low since rg_ch_ix
        if breakdown is True:
            df = calc_breakdown(
                rg_ch_data=fc_data.iloc[-1],
                latest_sw_discovery=latest_swing_data.end
            )
    #             breakdown = False
    #             breakout = True

        # try:
        #     _fc_data = fc_data.iloc[2:]
        #     df[rg_ch] = np.nan
        #     floors_data = _fc_data.loc[_fc_data.type == 1]
        #     ceilings_data = _fc_data.loc[_fc_data.type == -1]
        #     df.loc[floors_data.fc_date, flr] = floors_data.fc_val.values
        #     df.loc[ceilings_data.fc_date, clg] = ceilings_data.fc_val.values
        #     df.loc[_fc_data.rg_ch_date, rg_ch] = _fc_data.rg_ch_val.values
        #     df[rg_ch] = df[rg_ch].fillna(method="ffill")
        #     df[['hi3', 'lo3', flr, clg, rg_ch, 'close', 'rg']].plot(secondary_y='rg', style=['r.', 'g.', 'k^', 'kv'])
        #     pass
        # except:
        #     pass

    # no data excluding the initialized floor/ceiling
    if len(fc_data.iloc[2:]) == 0:
        raise NotEnoughDataError

    # POPULATE FLOOR,CEILING, RG CHANGE COLUMNS

    # remove init floor ceiling rows
    fc_data = fc_data.iloc[2:]
    floors_data = fc_data.loc[fc_data.type == 1]
    ceilings_data = fc_data.loc[fc_data.type == -1]

    df.loc[floors_data.fc_date, flr] = floors_data.fc_val.values
    df.loc[ceilings_data.fc_date, clg] = ceilings_data.fc_val.values
    df.loc[fc_data.rg_ch_date, rg_ch] = fc_data.rg_ch_val.values
    df[rg_ch] = df[rg_ch].fillna(method="ffill")

    # regime from last swing
    if latest_swing_data is not None:
        df.loc[latest_swing_data.end:, rg] = np.where(
            ceiling_found,  # if ceiling found, highest high since rg_ch_ix
            np.sign(df.loc[latest_swing_data.end:, _c].cummax() - fc_data.rg_ch_val.iloc[-1]),
            np.where(
                floor_found,  # if floor found, lowest low since rg_ch_ix
                np.sign(df.loc[latest_swing_data.end:, _c].cummin() - fc_data.rg_ch_val.iloc[-1]),
                # np.sign(df[swing_discovery_date:][_c].rolling(5).mean() - rg_ch_list[-1]),
                np.nan
            ),
        )
    df[rg] = df[rg].fillna(method="ffill")
    #     #     df[rg+'_no_fill'] = df[rg]
    return df


def find_fc(
        df,
        fc_ix: pd.Timestamp,
        price_col: str,
        extreme_func: str,
        stdev,
        sw_type: int,
        threshold: float,
        latest_swing: pd.Series,
) -> t.Dict:
    """
    tests to find a new fc between the last opposite fc and current swing.
    New fc found if the distance from the most extreme to the latest swing
    meets the minimum threshold

    if finding floor, Get min value between last ceiling and most recent swing low
    :param latest_swing:
    :param fc_ix:
    :param df:
    :param price_col:
    :param extreme_func:
    :param stdev:
    :param sw_type:
    :param threshold:
    :return:
    """
    res = {}

    # try again with next swing if fc > current swing
    # if fc_ix >= latest_swing.start:
    #     return res

    data_range = df.loc[fc_ix: latest_swing.start, price_col]
    # extreme_rows = data_range.loc[data_range == getattr(data_range, extreme_func)()]
    extreme_val = getattr(data_range, extreme_func)()
    extreme_idx = getattr(data_range, f'idx{extreme_func}')()
    # fc_val = extreme_rows.iloc[0]
    # fc_date = extreme_rows.index[0]
    fc_test = round((latest_swing.st_px - extreme_val) / stdev[latest_swing.start], 1)
    fc_test *= sw_type

    if fc_test >= threshold:
        res = {
            'test': fc_test,
            'fc_val': extreme_val,
            'fc_date': extreme_idx,
            'rg_ch_date': latest_swing.end,
            'rg_ch_val': latest_swing.st_px,
            'type': sw_type
        }
    return res


def hof_find_fc(df, price_col, extreme_func, stdev, sw_type, threshold):
    def _fc_found(fc_ix, latest_swing):
        return find_fc(df, fc_ix, price_col, extreme_func, stdev, sw_type, threshold, latest_swing)
    return _fc_found


def assign_retest_vals(c_data, retest_col, close_col):
    rt = c_data.copy()
    rt.loc[rt[retest_col].isna(), close_col] = np.nan
    rt = rt[close_col].copy()
    rt = rt.ffill()
    _cum_func = 'cummax' if 'lo' in retest_col else 'cummin'
    return getattr(rt, _cum_func)().ffill()


def normal_assign(c_data, retest_col, close_col):
    _cum_func = 'cummax' if 'lo' in retest_col else 'cummin'
    cd = c_data[close_col].copy()
    return getattr(cd, _cum_func)()


def fc_found(
        df,
        latest_hi_lo_sw_discovery,
        rg_data: pd.Series,
        cum_func: str,
        fc_type: int,
        retest: str,
        close_col='close',
        rg_col='rg',
):
    """
    set regime to where the newest swing was DISCOVERED

    """
    # close_data = df.loc[rg_data.rg_ch_date: latest_hi_lo_sw_discovery, close_col]
    # select close prices where retests have occurred
    close_data = df.loc[rg_data.rg_ch_date: latest_hi_lo_sw_discovery]
    # lo_retest = assign_retest_vals(close_data, 'lo1', close_col)
    # hi_retest = assign_retest_vals(close_data, 'hi1', close_col)
    # rt = assign_retest_vals(close_data, retest, close_col)

    rt = normal_assign(close_data, retest, close_col)
    df.loc[rg_data.rg_ch_date: latest_hi_lo_sw_discovery, rg_col] = np.sign(
        rt - rg_data.rg_ch_val
    )

    # 2. if price.cummax/cummin penetrates swing: regime turns bullish/bearish, breakout/breakdown
    # if retest occurs beyond the rg_ch level switch sides, but check for pullback/ trend resumption
    test_break = False
    if (df.loc[rg_data.rg_ch_date: latest_hi_lo_sw_discovery, rg_col] * fc_type > 0).any():
        # Boolean flags reset
        test_break = True

    return test_break, df


def hof_fc_found(df, cum_func, fc_type, retest, close_col='close', rg_col='rg'):
    def _fc_found(rg_ch_data, latest_hi_lo_sw_discovery):
        return fc_found(
            df=df,
            latest_hi_lo_sw_discovery=latest_hi_lo_sw_discovery,
            rg_data=rg_ch_data,
            cum_func=cum_func,
            fc_type=fc_type,
            close_col=close_col,
            rg_col=rg_col,
            retest=retest
        )
    return _fc_found


def hof_break_pullback(df, retest, extreme_idx_f, extreme_val_f):
    def _break_pullback(rg_ch_data, latest_sw_discovery):
        return break_pullback(
            df=df,
            rg_ch_data=rg_ch_data,
            latest_hi_lo_sw_discovery=latest_sw_discovery,
            extreme_idx_func=extreme_idx_f,
            extreme_val_func=extreme_val_f,
            retest=retest,
            rg_col='rg',
            close_col='close'
        )
    return _break_pullback


def break_pullback(
        df,
        rg_ch_data,
        latest_hi_lo_sw_discovery,
        extreme_idx_func: str,
        extreme_val_func: str,
        retest,
        rg_col='rg',
        close_col='close'
):
    # TODO, if retest pass, rg should be set starting with swing discovery date
    # TODO

    # data_range = df.loc[rg_ch_data.rg_ch_date: latest_hi_lo_sw_discovery, close_col]
    data_range = df.loc[rg_ch_data.rg_ch_date: latest_hi_lo_sw_discovery].copy()
    # break_extreme_date = getattr(data_range, extreme_idx_func)()

    # brkout_low = df[brkout_high_ix: latest_hi_lo_swing][close_col].cummin()
    # break_vals = df.loc[break_extreme_date: latest_hi_lo_sw_discovery, close_col]
    # break_vals = df.loc[break_extreme_date: latest_hi_lo_sw_discovery, close_col].ffill()

    diff = data_range.close - rg_ch_data.rg_ch_val
    data_range['retests'] = np.where(
        diff > 0,
        data_range.lo2,
        data_range.hi2
    )
    data_range['retests'] = data_range['retests'].ffill()
    # break_val = getattr(break_vals, extreme_val_func)()

    df.loc[rg_ch_data.rg_ch_date: latest_hi_lo_sw_discovery, rg_col] = np.sign(
        data_range['retests'] - rg_ch_data.rg_ch_val
    )
    return df
