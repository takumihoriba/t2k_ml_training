from analyze_output.analyze_ml_regression import analyze_ml_regression
from analyze_output.analyze_fitqun_regression import analyze_fitqun_regression

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

import numpy as np


class analysisResults():
     def __init__(self,settings):
        self.settings=settings
        self.global_perf = {}
        self.var_perf = {}
     def add_global_perf(self, name, axis, quantile, quantile_error, median, median_error):
        self.global_perf[name+self.settings.plotName] = (axis, [quantile, quantile_error, median, median_error])
     def get_global_perf(self, name, axis=None, value=None):
        if value is not None and axis is not None:
            axis_idx = self.global_perf[name+self.settings.plotName][0].index(axis)
            return self.global_perf[name+self.settings.plotName][1][self._global_value_idx(value)][axis_idx]
        elif value is not None:
           return self.global_perf[name+self.settings.plotName][1][self._global_value_idx(value)] 
        elif axis is not None:
            axis_idx = self.global_perf[name+self.settings.plotName][0].index(axis)
            return self.global_perf[name+self.settings.plotName][1][:,axis_idx]
        else:
            return self.global_perf[name+self.settings.plotName]
     def _global_value_idx(self, value):
        if "quantile" in value and "error" in value:
            return 1
        elif "quantile" in value:
            return 0
        elif "median" in value and "error" in value:
            return 2
        elif "median" in value:
            return 3

     def add_var_perf(self, name, dictionary):
            self.var_perf[name+self.settings.plotName] =  dictionary
     def get_var_perf(self, name, variable=None, axis=None, value=None):
        if value is not None and axis is not None:
            return np.array(self.var_perf[name+self.settings.plotName][variable][self._var_value_idx(value)][axis])
        elif value is not None:
           return self.var_perf[name+self.settings.plotName][variable][self._var_value_idx(value)] 
        else:
            return self.var_perf[name+self.settings.plotName][variable]
     def _var_value_idx(self, value):
        if "bins" in value:
            return 0
        if "quantile" in value and "error" in value:
            return 2
        elif "quantile" in value:
            return 1
        elif "median" in value and "error" in value:
            return 4
        elif "median" in value:
            return 3
     

def analyze_regression(settings):

    results = analysisResults(settings)

    if settings.doML:
        #vertex_axis_ml, quantile_lst_ml, quantile_error_lst_ml, median_lst_ml, median_error_lst_ml = analyze_ml_regression(settings)
        single_ml_analysis, multi_ml_analysis = analyze_ml_regression(settings) 
        #print(f"SINGLE ANALYSIS: {single_ml_analysis}")
        #print(f"MULTI ANALYSIS: {multi_ml_analysis}")
        results.add_global_perf("ML", single_ml_analysis[0], single_ml_analysis[1], single_ml_analysis[2], single_ml_analysis[3], single_ml_analysis[4])
        results.add_var_perf("ML",multi_ml_analysis)

    #print(f"Quantile: {results.get_var_perf('ML', variable='ve', axis='Angle', value='quantile')}")
    #print(f"Median: {results.get_var_perf('ML', variable='ve', axis='Angle', value='median')}")

    if settings.doFiTQun:
        #vertex_axis_fq, quantile_lst_fq, quantile_error_lst_fq, median_lst_fq, median_error_lst_fq = analyze_fitqun_regression(settings)
        single_fq_analysis, multi_fq_analysis = analyze_fitqun_regression(settings) 
        results.add_global_perf("fitqun", single_fq_analysis[0], single_fq_analysis[1], single_fq_analysis[2], single_fq_analysis[3], single_fq_analysis[4])
        results.add_var_perf("fitqun",multi_fq_analysis)
        #results.add_global_perf("fitqun", vertex_axis_fq, quantile_lst_fq, quantile_error_lst_fq, median_lst_fq, median_error_lst_fq)
    
    #print(results.get_global_perf("fitqun", axis="Angle", value="quantile"))
    #print(results.get_global_perf("ML", axis="Angle", value="quantile"))

    if "directions" in settings.target:
        print(f"Directions; ML; Angle")
        print(f"Resolution {results.get_global_perf('ML', axis='Angle', value='quantile')} ({results.get_global_perf('ML', axis='Angle', value='quantile error')})")
        print(f"Bias {results.get_global_perf('ML', axis='Angle', value='median')} ({results.get_global_perf('ML', axis='Angle', value='median error')})")
        print(f"Directions; fiTQun; Angle")
        print(f"Resolution {results.get_global_perf('fitqun', axis='Angle', value='quantile')} ({results.get_global_perf('fitqun', axis='Angle', value='quantile error')})")
        print(f"Bias {results.get_global_perf('fitqun', axis='Angle', value='median')} ({results.get_global_perf('fitqun', axis='Angle', value='median error')})")
        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Angle', value='bins'), results.get_var_perf("ML", variable='ve', axis='Angle', value='quantile'),
                        results.get_var_perf("ML", variable='ve', axis='Angle', value='quantile error'), results.get_var_perf("fitqun", variable='ve', axis='Angle', value='quantile'),
                        results.get_var_perf("fitqun", variable='ve', axis='Angle', value='quantile error'), "Visible Energy [MeV]", "Angle Resolution [deg]", "ve_angle_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Angle', value='bins'), results.get_var_perf("ML", variable='towall', axis='Angle', value='quantile'),
                        results.get_var_perf("ML", variable='towall', axis='Angle', value='quantile error'), results.get_var_perf("fitqun", variable='towall', axis='Angle', value='quantile'),
                        results.get_var_perf("fitqun", variable='towall', axis='Angle', value='quantile error'), "Towall [cm]", "Angle Resolution [deg]", "towall_angle_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Angle', value='bins'), results.get_var_perf("ML", variable='ve', axis='Angle', value='median'),
                        results.get_var_perf("ML", variable='ve', axis='Angle', value='median error'), results.get_var_perf("fitqun", variable='ve', axis='Angle', value='median'),
                        results.get_var_perf("fitqun", variable='ve', axis='Angle', value='median error'), "Visible Energy [MeV]", "Angle Bias [deg]", "ve_angle_ml_fq_median", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Angle', value='bins'), results.get_var_perf("ML", variable='towall', axis='Angle', value='median'),
                        results.get_var_perf("ML", variable='towall', axis='Angle', value='median error'), results.get_var_perf("fitqun", variable='towall', axis='Angle', value='median'),
                        results.get_var_perf("fitqun", variable='towall', axis='Angle', value='median error'), "Towall [cm]", "Angle Bias [deg]", "towall_angle_ml_fq_median", settings)

    if "momenta" in settings.target:
        print(f"Directions; ML; Global")
        print(f"Resolution {100*results.get_global_perf('ML', axis='Global', value='quantile')} ({100*results.get_global_perf('ML', axis='Global', value='quantile error')})")
        print(f"Bias {100*results.get_global_perf('ML', axis='Global', value='median')} ({100*results.get_global_perf('ML', axis='Global', value='median error')})")
        print(f"Directions; fiTQun; Global")
        print(f"Resolution {100*results.get_global_perf('fitqun', axis='Global', value='quantile')} ({100*results.get_global_perf('fitqun', axis='Global', value='quantile error')})")
        print(f"Bias {100*results.get_global_perf('fitqun', axis='Global', value='median')} ({100*results.get_global_perf('fitqun', axis='Global', value='median error')})")
        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='ve', axis='Global', value='quantile'),
                        100*results.get_var_perf("ML", variable='ve', axis='Global', value='quantile error'), 100*results.get_var_perf("fitqun", variable='ve', axis='Global', value='quantile'),
                        100*results.get_var_perf("fitqun", variable='ve', axis='Global', value='quantile error'), "Visible Energy [MeV]", " Momentum Resolution (%)", "ve_mom_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='towall', axis='Global', value='quantile'),
                        100*results.get_var_perf("ML", variable='towall', axis='Global', value='quantile error'), 100*results.get_var_perf("fitqun", variable='towall', axis='Global', value='quantile'),
                        100*results.get_var_perf("fitqun", variable='towall', axis='Global', value='quantile error'), "Towall [cm]", "Momentum Resolution [%]", "towall_mom_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='ve', axis='Global', value='median'),
                        100*results.get_var_perf("ML", variable='ve', axis='Global', value='median error'), 100*results.get_var_perf("fitqun", variable='ve', axis='Global', value='median'),
                        100*results.get_var_perf("fitqun", variable='ve', axis='Global', value='median error'), "Visible Energy [MeV]", "Momentum Bias [%]", "ve_mom_ml_fq_median", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='towall', axis='Global', value='median'),
                        100*results.get_var_perf("ML", variable='towall', axis='Global', value='median error'), 100*results.get_var_perf("fitqun", variable='towall', axis='Global', value='median'),
                        100*results.get_var_perf("fitqun", variable='towall', axis='Global', value='median error'), "Towall [cm]", "Momentum Bias [%]", "towall_mom_ml_fq_median", settings)

    if "positions" in settings.target:
        print(f"Positions; ML; Global")
        print(f"Resolution {results.get_global_perf('ML', axis='Global', value='quantile')} ({results.get_global_perf('ML', axis='Global', value='quantile error')})")
        print(f"Bias {results.get_global_perf('ML', axis='Global', value='median')} ({results.get_global_perf('ML', axis='Global', value='median error')})")
        print(f"Directions; fiTQun; Global")
        print(f"Resolution {results.get_global_perf('fitqun', axis='Global', value='quantile')} ({results.get_global_perf('fitqun', axis='Global', value='quantile error')})")
        print(f"Bias {results.get_global_perf('fitqun', axis='Global', value='median')} ({results.get_global_perf('fitqun', axis='Global', value='median error')})")

        print(f"Positions; ML; Transverse")
        print(f"Resolution {results.get_global_perf('ML', axis='Transverse', value='quantile')} ({results.get_global_perf('ML', axis='Transverse', value='quantile error')})")
        print(f"Bias {results.get_global_perf('ML', axis='Transverse', value='median')} ({results.get_global_perf('ML', axis='Transverse', value='median error')})")
        print(f"Directions; fiTQun; Transverse")
        print(f"Resolution {results.get_global_perf('fitqun', axis='Transverse', value='quantile')} ({results.get_global_perf('fitqun', axis='Transverse', value='quantile error')})")
        print(f"Bias {results.get_global_perf('fitqun', axis='Transverse', value='median')} ({results.get_global_perf('fitqun', axis='Transverse', value='median error')})")

        print(f"Positions; ML; Longitudinal")
        print(f"Resolution {results.get_global_perf('ML', axis='Longitudinal', value='quantile')} ({results.get_global_perf('ML', axis='Longitudinal', value='quantile error')})")
        print(f"Bias {results.get_global_perf('ML', axis='Longitudinal', value='median')} ({results.get_global_perf('ML', axis='Longitudinal', value='median error')})")
        print(f"Directions; fiTQun; Longitudinal")
        print(f"Resolution {results.get_global_perf('fitqun', axis='Longitudinal', value='quantile')} ({results.get_global_perf('fitqun', axis='Longitudinal', value='quantile error')})")
        print(f"Bias {results.get_global_perf('fitqun', axis='Longitudinal', value='median')} ({results.get_global_perf('fitqun', axis='Longitudinal', value='median error')})")

        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), results.get_var_perf("ML", variable='ve', axis='Global', value='quantile'),
                        results.get_var_perf("ML", variable='ve', axis='Global', value='quantile error'), results.get_var_perf("fitqun", variable='ve', axis='Global', value='quantile'),
                        results.get_var_perf("fitqun", variable='ve', axis='Global', value='quantile error'), "Visible Energy [MeV]", "Global Resolution [cm]", "ve_global_pos_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), results.get_var_perf("ML", variable='towall', axis='Global', value='quantile'),
                        results.get_var_perf("ML", variable='towall', axis='Global', value='quantile error'), results.get_var_perf("fitqun", variable='towall', axis='Global', value='quantile'),
                        results.get_var_perf("fitqun", variable='towall', axis='Global', value='quantile error'), "Towall [cm]", "Global Resolution [cm]", "towall_global_pos_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='ve', axis='Transverse', value='quantile'),
                        results.get_var_perf("ML", variable='ve', axis='Transverse', value='quantile error'), results.get_var_perf("fitqun", variable='ve', axis='Transverse', value='quantile'),
                        results.get_var_perf("fitqun", variable='ve', axis='Transverse', value='quantile error'), "Visible Energy [MeV]", "Transverse Resolution [cm]", "ve_transverse_pos_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='towall', axis='Transverse', value='quantile'),
                        results.get_var_perf("ML", variable='towall', axis='Transverse', value='quantile error'), results.get_var_perf("fitqun", variable='towall', axis='Transverse', value='quantile'),
                        results.get_var_perf("fitqun", variable='towall', axis='Transverse', value='quantile error'), "Towall [cm]", "Transverse Resolution [cm]", "towall_transverse_pos_ml_fq_res", settings)
                        
        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='quantile'),
                        results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='quantile error'), results.get_var_perf("fitqun", variable='ve', axis='Longitudinal', value='quantile'),
                        results.get_var_perf("fitqun", variable='ve', axis='Longitudinal', value='quantile error'), "Visible Energy [MeV]", "Longitudinal Resolution [cm]", "ve_longitudinal_pos_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='quantile'),
                        results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='quantile error'), results.get_var_perf("fitqun", variable='towall', axis='Longitudinal', value='quantile'),
                        results.get_var_perf("fitqun", variable='towall', axis='Longitudinal', value='quantile error'), "Towall [cm]", "Longitudinal Resolution [cm]", "towall_longitudinal_pos_ml_fq_res", settings)

        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), results.get_var_perf("ML", variable='ve', axis='Global', value='median'),
                        results.get_var_perf("ML", variable='ve', axis='Global', value='median error'), results.get_var_perf("fitqun", variable='ve', axis='Global', value='median'),
                        results.get_var_perf("fitqun", variable='ve', axis='Global', value='median error'), "Visible Energy [MeV]", "Global Bias [cm]", "ve_global_pos_ml_fq_median", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), results.get_var_perf("ML", variable='towall', axis='Global', value='median'),
                        results.get_var_perf("ML", variable='towall', axis='Global', value='median error'), results.get_var_perf("fitqun", variable='towall', axis='Global', value='median'),
                        results.get_var_perf("fitqun", variable='towall', axis='Global', value='median error'), "Towall [cm]", "Global Bias [cm]", "towall_global_pos_ml_fq_median", settings)

        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='ve', axis='Transverse', value='median'),
                        results.get_var_perf("ML", variable='ve', axis='Transverse', value='median error'), results.get_var_perf("fitqun", variable='ve', axis='Transverse', value='median'),
                        results.get_var_perf("fitqun", variable='ve', axis='Transverse', value='median error'), "Visible Energy [MeV]", "Transverse Bias [cm]", "ve_transverse_pos_ml_fq_median", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='towall', axis='Transverse', value='median'),
                        results.get_var_perf("ML", variable='towall', axis='Transverse', value='median error'), results.get_var_perf("fitqun", variable='towall', axis='Transverse', value='median'),
                        results.get_var_perf("fitqun", variable='towall', axis='Transverse', value='median error'), "Towall [cm]", "Transverse Bias [cm]", "towall_transverse_pos_ml_fq_median", settings)
                        
        plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='median'),
                        results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='median error'), results.get_var_perf("fitqun", variable='ve', axis='Longitudinal', value='median'),
                        results.get_var_perf("fitqun", variable='ve', axis='Longitudinal', value='median error'), "Visible Energy [MeV]", "Longitudinal Bias [cm]", "ve_longitudinal_pos_ml_fq_median", settings)

        plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='median'),
                        results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='median error'), results.get_var_perf("fitqun", variable='towall', axis='Longitudinal', value='median'),
                        results.get_var_perf("fitqun", variable='towall', axis='Longitudinal', value='median error'), "Towall [cm]", "Longitudinal Bias [cm]", "towall_longitudinal_pos_ml_fq_median", settings)


def plot_reg_results(x, ml, ml_error, fitqun, fitqun_error, xlabel, ylabel, name, settings):
    plt.errorbar(x, ml, ml_error, label="ML", ls='none', marker='o')
    plt.errorbar(x, fitqun, fitqun_error, label="fiTQun", ls='none', marker='^')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(settings.outputPlotPath+'/'+name+'.png', bbox_inches='tight')
    plt.clf()

    