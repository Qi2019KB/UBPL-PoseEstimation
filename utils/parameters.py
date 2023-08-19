import numpy as np


def update_ema_variables(model, ema_model, args):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (args.epo + 1), args.ema_decay)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# min --> max
def consWeight_increase(epo, args):
    return _value_increase(epo, args.consWeight_max, args.consWeight_min, args.consWeight_rampup)


# min --> max
def pseudoWeight_increase(epo, args):
    return _value_increase(epo, args.pseudoWeight_max, args.pseudoWeight_min, args.pseudoWeight_rampup)


# min --> max
def pseudoSelRate_increase(epo, args):
    return _value_increase(epo, args.pseudoSelRate_max, args.pseudoSelRate_min, args.pseudoSelRate_rampup)


# min --> max
def reliablePCT_increase(epo, args):
    return _value_increase(epo, args.reliablePCT_max, args.reliablePCT_min, args.reliablePCT_rampup)


# min --> max
def uncPCT_increase(epo, args):
    return _value_increase(epo, args.uncPCT_max, args.uncPCT_min, args.uncPCT_rampup)


def brWeight_increase(epo, args):
    return _value_increase(epo, args.brWeight_max, args.brWeight_min, args.brWeight_rampup)


# max --> min
def uncPCT_decrease(epo, args):
    return _value_decrease(epo, args.uncPCT_max, args.uncPCT_min, args.uncPCT_rampup)


# min --> max
def scoreThr_increase(epo, args):
    return _value_increase(epo, args.scoreThr_max, args.scoreThr_min, args.scoreThr_rampup)


def FDLWeight_increase(epo, args):
    return _value_increase(epo, args.FDLWeight_max, args.FDLWeight_min, args.FDLWeight_rampup)


def FILWeight_decrease(epo, args):
    return _value_decrease(epo, args.FILWeight_max, args.FILWeight_min, args.FILWeight_rampup)


def FDLWeight_decrease(epo, args):
    return _value_decrease(epo, args.FDLWeight_max, args.FDLWeight_min, args.FDLWeight_rampup)

def FDLWeight_Step(epo, stages, values, epochs):
    if stages[0] > 0:
        stages = [0] + stages
        values = [0.0] + values

    if stages[-1] < epochs:
        stages = stages + [500]
        values = values + [0.0]


    inIdx = 0
    for sIdx, stage in enumerate(stages):
        if epo >= stage: inIdx = sIdx

    minValue = values[inIdx]
    maxValue = values[inIdx+1]
    rampupValue = stages[inIdx+1] - stages[inIdx]
    epoValue = epo - stages[inIdx]
    if minValue <= maxValue:
        val = _value_increase(epoValue, maxValue, minValue, rampupValue)
    else:
        val = _value_decrease(epoValue, minValue, maxValue, rampupValue)
    return val


def FDLWeight_CAWR(epo, stages, startValues, min_value):
    stages_plus = [0] + stages

    inIdx = 0
    for sIdx, stage in enumerate(stages_plus):
        if epo >= stage: inIdx = sIdx

    maxValue = startValues[inIdx]
    minValue = min_value
    rampup = stages_plus[inIdx+1] - stages_plus[inIdx]
    epoValue = (epo - stages_plus[inIdx]) if inIdx > 0 else epo
    return _value_decrease(epoValue, maxValue, minValue, rampup)


def _value_increase(epo, maxValue, minValue, rampup):
    return minValue + (maxValue - minValue) * _sigmoid_rampup(epo, rampup)


def _value_decrease(epo, maxValue, minValue, rampup):
    return minValue + (maxValue - minValue) * (1.0 - _sigmoid_rampup(epo, rampup))


def _sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))