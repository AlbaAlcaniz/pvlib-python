import numpy as np


def sizes_and_weights():
    """SIZES_AND_WEIGHTS Calculate necessary sizes and weights of the pontoon
    %
    % Syntax:
    %   [m_tot, wid, leng, thick_block, area_pontoon] = sizes_and_weights()
    %
    % Description:
    %   This function provides the mass and dimensions of the floater
    %   considered in the simulations. For this purpose, the type of floater
    %   and type and number of PV modules need to be considered. Other
    %   distances such as walkability and maintenance are also considered. A
    %   floatibility check is performed to ensure that the floater will
    %   actually float This function is specific to each floater. The floater
    %   is assumed to be a rigid rectangular cuboid.
    %
    % Outputs:
    %   m_tot: total mass of the floater [kg]
    %   wid: width of the floater [m]
    %   leng: length of the floater [m]
    %   thick_block: thickness of the floater [m]
    %   area_pontoon: area of the floater (width*length) [m^2]

        Returns:
            _type_: _description_
    """

    # Sunnydock block characteristics
    # Dimensions [m]
    thick_block = 0.4
    len_block = 0.5
    wid_block = 0.5
    area_block = len_block * wid_block
    # Weight [kg]
    mass_block = 6.5
    # Flotation [kg]
    flotation_block = 87.5

    # Calculation of the number of modules limited by the inverter
    # PV module LG400N2W-A5 characteristics
    # Dimensions [m]
    len_pv = 1.024
    wid_pv = 2.024
    # Mass [kg]
    mass_pv = 21.7
    # Electrical characteristics: power [W] and OC voltage [V]
    P_max = 400
    Voc = 49.3

    # SunnyBoy inverter characteristics
    # Electrical characteristics: power [W] and OC voltage [V]
    Inverter_power = 5000
    Inverter_voltage = 600
    # Mass [kg]
    mass_inverter = 9.2

    # Max number of modules in a string
    n_string = min(np.floor(Inverter_power / P_max), np.floor(Inverter_voltage / Voc))

    # Other distances [m]
    # Walkability
    d_rows = 0.7
    # Distance between modules
    d_columns = 0.15
    # Distance for maintanance operation + all elements allocation
    maintenance_space = 5

    # Final characteristics
    # Desired pontoon length
    wid = np.ceil(n_string * (wid_pv + d_columns)) + maintenance_space
    leng = 32
    # Pontoon area
    area_pontoon = leng * wid
    # Number of modules
    n_rows = np.floor(leng / ((len_pv + d_rows)))
    n_modules = n_rows * n_string

    # Mass calculations [kg]
    m_pv = mass_pv * n_modules
    m_pontoon = (area_pontoon / area_block) * mass_block

    # Anchoring
    h_water = 40
    l_chain = max(wid * np.sin(np.deg2rad(60)), leng * np.sin(np.deg2rad(60))) + h_water
    density_chains = 4
    mass_anchor_clips = 10 * 4
    m_anchoring = 4 * l_chain * density_chains + mass_anchor_clips

    # Balance Of System
    l_cables = (wid - maintenance_space) * n_rows + leng
    density_cables = 0.086
    m_cables = l_cables * density_cables
    m_inverter = mass_inverter * n_rows
    wid_mounting = wid - maintenance_space
    h_mounting = 0.3
    len_mounting = 0.2
    density_304 = 85000
    m_mountframe = 0.589 * (len_mounting * h_mounting) * wid_mounting * density_304
    m_mountframe = 1.08 * m_mountframe  # to consider clips and bolts
    m_BOS = m_cables + m_inverter + m_mountframe

    # Total mass
    m_tot = m_pontoon + m_pv + m_BOS + m_anchoring

    # Flotation check
    n_block = area_pontoon / area_block
    flotation = n_block * flotation_block

    # Verify if pontoon can withstand the weight
    if m_tot < flotation:
        print("Pontoon can withstand the weight")
    else:
        print("Pontoon cannot withstand the weight")

    return m_tot, wid, leng, thick_block, area_pontoon
