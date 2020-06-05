# A dictionary for unfolding observables and their configurations
observable_dict = {}

# ttbar mass
observable_dict['mtt'] = {
    'branch_det': 'mttReco', 'branch_mc': 'mttTrue',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (0, 1500), 'ylim': (0, 1.),
    'xlabel': 'm_ttbar', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# ttbar pt
observable_dict['ptt'] = {
    'branch_det': 'pttReco', 'branch_mc': 'pttTrue',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (0, 800), 'ylim': (0, 1.),
    'xlabel': 'pt_ttbar', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# ttbar rapidity
observable_dict['ytt'] = {
    'branch_det': 'yttReco', 'branch_mc': 'yttTrue',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-3, -3), 'ylim': (0, 1.),
    'xlabel': 'y_ttbar', 'ylabel': '',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# absolute value of ttbar rapidity in the their center of mass frame
observable_dict['ystar'] = {
    'branch_det': 'ystarReco', 'branch_mc': 'ystarTrue',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (0, 3), 'ylim': (0, 1.),
    'xlabel': 'ystar', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# rapidity boost of the ttbar system
observable_dict['yboost'] = {
    'branch_det': 'yboostReco', 'branch_mc': 'yboostTrue',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-3, 3), 'ylim': (0, 1.),
    'xlabel': 'yboost', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# absolution value of differences in phi between the two tops
observable_dict['dphi'] = {
    'branch_det': 'dphiReco', 'branch_mc': 'dphiTrue',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (0, 3.2), 'ylim': (0, 1.),
    'xlabel': 'dphi', 'ylabel': '',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# scalar sum of pt of the two tops
observable_dict['Ht'] = {
    'branch_det': 'HtReco', 'branch_mc': 'HtTrue',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (0, 1000), 'ylim': (0, 1.),
    'xlabel': 'Ht', 'ylabel': '',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# hadronic top
# pt
observable_dict['th_pt'] = {
    'branch_det': 'th_pt', 'branch_mc': 'th_pt_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (0, 500), 'ylim': (0, 1.),
    'xlabel': 'pt', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# eta
observable_dict['th_eta'] = {
    'branch_det': 'th_eta', 'branch_mc': 'th_eta_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-5, 5), 'ylim': (0, 1.),
    'xlabel': 'eta', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# rapidity
observable_dict['th_y'] = {
    'branch_det': 'th_y', 'branch_mc': 'th_y_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-3, 3), 'ylim': (0, 1.),
    'xlabel': 'y', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# phi
observable_dict['th_phi'] = {
    'branch_det': 'th_phi', 'branch_mc': 'th_phi_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-3.2, 3.2), 'ylim': (0, 1.),
    'xlabel': 'phi', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# mass
observable_dict['th_m'] = {
    'branch_det': 'th_m', 'branch_mc': 'th_m_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (100, 240), 'ylim': (0, 1.),
    'xlabel': 'mass', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# out-of-plane momentum
observable_dict['th_pout'] = {
    'branch_det': 'th_pout', 'branch_mc': 'th_pout_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-250, 250), 'ylim': (0, 1.),
    'xlabel': 'p_out', 'ylabel': '',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# leptonic top
# pt
observable_dict['tl_pt'] = {
    'branch_det': 'tl_pt', 'branch_mc': 'tl_pt_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (0, 500), 'ylim': (0, 1.),
    'xlabel': 'pt', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# eta
observable_dict['tl_eta'] = {
    'branch_det': 'tl_eta', 'branch_mc': 'tl_eta_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-5, 5), 'ylim': (0, 1.),
    'xlabel': 'eta', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# rapidity
observable_dict['tl_y'] = {
    'branch_det': 'tl_y', 'branch_mc': 'tl_y_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-3, 3), 'ylim': (0, 1.),
    'xlabel': 'y', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# phi
observable_dict['tl_phi'] = {
    'branch_det': 'tl_phi', 'branch_mc': 'tl_phi_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-3.2, 3.2), 'ylim': (0, 1.),
    'xlabel': 'phi', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# mass
observable_dict['tl_m'] = {
    'branch_det': 'tl_m', 'branch_mc': 'tl_m_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (100, 240), 'ylim': (0, 1.),
    'xlabel': 'mass', 'ylabel':'',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}

# out-of-plane momentum
observable_dict['tl_pout'] = {
    'branch_det': 'tl_pout', 'branch_mc': 'tl_pout_MC',
    'nbins_det': 50, 'nbins_mc': 50,
    'xlim': (-250, 250), 'ylim': (0, 1.),
    'xlabel': 'p_out', 'ylabel': '',
    #'stamp_xy': (0.41, 0.92),
    #'legend_loc': 'upper left', 'legend_ncol': 1,
}
