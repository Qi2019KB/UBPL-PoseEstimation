# -*- coding: utf-8 -*-
from projects.supervised import exec as Supervised
from projects.MT import exec as MT
from projects.MT_UBPL import exec as MT_UBPL
from projects.DualPose_UBPL import exec as DualPose_UBPL


def exec_home():
    for dataCount in [["Mouse", 100, 0.3], ["Mouse", 200, 0.15],
                      ["FLIC", 100, 0.3], ["FLIC", 200, 0.15],
                      ["LSP", 500, 0.2], ["LSP", 500, 0.4]]:
        dataSource, trainCount, rate = dataCount

        Supervised("Supervised", {"dataSource": dataSource, "trainCount": trainCount, "labelRatio": rate})

        MT("MT", {"dataSource": dataSource, "trainCount": trainCount, "labelRatio": rate})

        MT_UBPL("MT_UBPL", {"dataSource": dataSource, "trainCount": trainCount, "labelRatio": rate,
                            "FDLWeight_max": 1.0, "FDLWeight_min": 1.0, "useEnsemblePseudo": True})

        DualPose_UBPL("DualPose", {"dataSource": dataSource, "trainCount": trainCount, "labelRatio": rate,
                                   "FDLWeight_max": 0.0, "FDLWeight_min": 0.0, "useEnsemblePseudo": False})

        DualPose_UBPL("DualPose_UBPL", {"dataSource": dataSource, "trainCount": trainCount, "labelRatio": rate,
                                        "FDLWeight_max": 1.0, "FDLWeight_min": 1.0, "useEnsemblePseudo": True})
    pass


if __name__ == "__main__":
    exec_home()
