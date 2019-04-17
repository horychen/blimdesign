#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
# Design Specification
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
p = 1
spec = pyrhonen_procedure_as_function.desgin_specification(
        PS_or_SC = True, # Pole Specific or Squirrel Cage
        DPNV_or_SEPA = True, # Dual purpose no voltage or Separate winding
        p = p,
        ps = 2 if p==1 else 1,
        mec_power = 100e3, # kW
        ExcitationFreq = 880, # Hz
        VoltageRating = 480, # Vrms (line-to-line, Wye-Connect)
        TangentialStress = 12000, # Pa
        Qs = 24,
        Qr = 16,
        Js = 3.7e6, # Arms/m^2
        Jr = 7.25e6, #7.5e6, #6.575e6, # Arms/m^2
        Steel = 'M19Gauge29', # Arnon-7
        lamination_stacking_factor_kFe = 0.95, # from http://www.femm.info/wiki/spmloss # 0.91 for Arnon
        Coil = 'Cu',
        space_factor_kCu = 0.5, # Stator slot fill/packign factor
        Conductor = 'Cu',
        space_factor_kAl = 1.0, # Rotor slot fill/packing factor
        Temperature = 75, # deg Celsius
        stator_tooth_flux_density_B_ds = 1.4, # Tesla
        rotor_tooth_flux_density_B_dr  = 1.5, # Tesla
        stator_yoke_flux_density_Bys = 1.2, # Tesla
        rotor_yoke_flux_density_Byr  = 1.1 + 0.3 if p==1 else 1.1, # Tesla
        guess_air_gap_flux_density = 0.8, # 0.8, # Tesla | 0.7 ~ 0.9 | Table 6.3
        guess_efficiency = 0.95,
        guess_power_factor = 0.7,
        debug_or_release = True, # 如果是debug，数据库里有记录就删掉重新跑；如果release且有记录，那就报错。=debug_or_release = True # 如果是debug，数据库里有记录就删掉重新跑；如果release且有记录，那就报错。
        bool_skew_stator = None,
        bool_skew_rotor = None,
)