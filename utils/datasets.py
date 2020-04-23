# Only good for UCR Time Series Archive 2018. Will not work for 2015 version.

def nb_dims(dataset):
    return 1

def nb_classes(dataset):
    if dataset == "FiftyWords":
        return 50 #270
    if dataset == "Adiac":
        return 37 #176
    if dataset == "ArrowHead":
        return 3 #251
    if dataset == "Beef":
        return 5 #470
    if dataset == "BeetleFly":
        return 2 #512
    if dataset == "BirdChicken":
        return 2 #512
    if dataset == "Car":
        return 4 #577
    if dataset == "CBF":
        return 3 #128
    if dataset == "ChlorineConcentration":
        return 3 #166
    if dataset == "CinCECGTorso":
        return 4 #1639
    if dataset == "Coffee":
        return 2 #286
    if dataset == "Computers":
        return 2 #720
    if dataset == "CricketX":
        return 12 #300
    if dataset == "CricketY":
        return 12 #300
    if dataset == "CricketZ":
        return 12 #300
    if dataset == "DiatomSizeReduction":
        return 4 #345
    if dataset == "DistalPhalanxOutlineAgeGroup":
        return 3 #80
    if dataset == "DistalPhalanxOutlineCorrect":
        return 2 #80
    if dataset == "DistalPhalanxTW":
        return 6 #80
    if dataset == "Earthquakes":
        return 2 #512
    if dataset == "ECG200":
        return 2 #96
    if dataset == "ECG5000":
        return 5 #140
    if dataset == "ECGFiveDays":
        return 2 #136
    if dataset == "ElectricDevices":
        return 7 #96
    if dataset == "FaceAll":
        return 14 # 131
    if dataset == "FaceFour":
        return 4 # 350
    if dataset == "FacesUCR":
        return 14 # 131
    if dataset == "Fish":
        return 7 # 463
    if dataset == "FordA":
        return 2 #500
    if dataset == "FordB":
        return 2 # 500
    if dataset == "GunPoint":
        return 2 # 150
    if dataset == "Ham":
        return 2 # 431
    if dataset == "HandOutlines":
        return 2 # 2709
    if dataset == "Haptics":
        return 5 # 1092
    if dataset == "Herring":
        return 2 # 512
    if dataset == "InlineSkate":
        return 7 # 1882
    if dataset == "InsectWingbeatSound":
        return 11 # 256
    if dataset == "ItalyPowerDemand":
        return 2 # 24
    if dataset == "LargeKitchenAppliances":
        return 3 # 720
    if dataset == "Lightning2":
        return 2 # 637
    if dataset == "Lightning7":
        return 7 # 319
    if dataset == "Mallat":
        return 8 # 1024
    if dataset == "Meat":
        return 3 # 448
    if dataset == "MedicalImages":
        return 10 # 99
    if dataset == "MiddlePhalanxOutlineAgeGroup":
        return 3 #80
    if dataset == "MiddlePhalanxOutlineCorrect":
        return 2 #80
    if dataset == "MiddlePhalanxTW":
        return 6 #80
    if dataset == "MoteStrain":
        return 2 #84
    if dataset == "NonInvasiveFetalECGThorax1":
        return 42 #750
    if dataset == "NonInvasiveFetalECGThorax2":
        return 42 #750
    if dataset == "OliveOil":
        return 4 #570
    if dataset == "OSULeaf":
        return 6 #427
    if dataset == "PhalangesOutlinesCorrect":
        return 2 #80
    if dataset == "Phoneme":
        return 39 #1024
    if dataset == "Plane":
        return 7 #144
    if dataset == "ProximalPhalanxOutlineAgeGroup":
        return 3 #80
    if dataset == "ProximalPhalanxOutlineCorrect":
        return 2 #80
    if dataset == "ProximalPhalanxTW":
        return 6 #80
    if dataset == "RefrigerationDevices":
        return 3 #720
    if dataset == "ScreenType":
        return 3 #720
    if dataset == "ShapeletSim":
        return 2 #500
    if dataset == "ShapesAll":
        return 60 # 512
    if dataset == "SmallKitchenAppliances":
        return 3 #720
    if dataset == "SonyAIBORobotSurface2":
        return 2 #65
    if dataset == "SonyAIBORobotSurface1":
        return 2 #70
    if dataset == "StarLightCurves":
        return 3 #1024
    if dataset == "Strawberry":
        return 2 #235
    if dataset == "SwedishLeaf":
        return 15 # 128
    if dataset == "Symbols":
        return 6 #398
    if dataset == "SyntheticControl":
        return 6 #60
    if dataset == "ToeSegmentation1":
        return 2 #277
    if dataset == "ToeSegmentation2":
        return 2 #343
    if dataset == "Trace":
        return 4 #275
    if dataset == "TwoLeadECG":
        return 2 #82
    if dataset == "TwoPatterns":
        return 4 #128
    if dataset == "UWaveGestureLibraryX":
        return 8 # 315
    if dataset == "UWaveGestureLibraryY":
        return 8 # 315
    if dataset == "UWaveGestureLibraryZ":
        return 8 # 315
    if dataset == "UWaveGestureLibraryAll":
        return 8 # 945
    if dataset == "Wafer":
        return 2 #152
    if dataset == "Wine":
        return 2 #234
    if dataset == "WordsSynonyms":
        return 25 #270
    if dataset == "Worms":
        return 5 #900
    if dataset == "WormsTwoClass":
        return 2 #900
    if dataset == "Yoga":
        return 2 #426
    
    if dataset == "ACSF1":
        return 10
    if dataset == "AllGestureWiimoteX":
        return 10
    if dataset == "AllGestureWiimoteY":
        return 10
    if dataset == "AllGestureWiimoteZ":
        return 10
    if dataset == "BME":
        return 3
    if dataset == "Chinatown":
        return 2
    if dataset == "Crop":
        return 24
    if dataset == "DodgerLoopDay":
        return 7
    if dataset == "DodgerLoopGame":
        return 2
    if dataset == "DodgerLoopWeekend":
        return 2
    if dataset == "EOGHorizontalSignal":
        return 12
    if dataset == "EOGVerticalSignal":
        return 12
    if dataset == "EthanolLevel":
        return 4
    if dataset == "FreezerRegularTrain":
        return 2
    if dataset == "FreezerSmallTrain":
        return 2
    if dataset == "Fungi":
        return 18
    if dataset == "GestureMidAirD1":
        return 26
    if dataset == "GestureMidAirD2":
        return 26
    if dataset == "GestureMidAirD3":
        return 26
    if dataset == "GesturePebbleZ1":
        return 6
    if dataset == "GesturePebbleZ2":
        return 6
    if dataset == "GunPointAgeSpan":
        return 2
    if dataset == "GunPointMaleVersusFemale":
        return 2
    if dataset == "GunPointOldVersusYoung":
        return 2
    if dataset == "HouseTwenty":
        return 2
    if dataset == "InsectEPGRegularTrain":
        return 3
    if dataset == "InsectEPGSmallTrain":
        return 3
    if dataset == "MelbournePedestrian":
        return 10
    if dataset == "MixedShapesRegularTrain":
        return 5
    if dataset == "MixedShapesSmallTrain":
        return 5
    if dataset == "PickupGestureWiimoteZ":
        return 10
    if dataset == "PigAirwayPressure":
        return 52
    if dataset == "PigArtPressure":
        return 52
    if dataset == "PigCVP":
        return 52
    if dataset == "PLAID":
        return 11
    if dataset == "PowerCons":
        return 2
    if dataset == "Rock":
        return 4
    if dataset == "SemgHandGenderCh2":
        return 2
    if dataset == "SemgHandMovementCh2":
        return 6
    if dataset == "SemgHandSubjectCh2":
        return 5
    if dataset == "ShakeGestureWiimoteZ":
        return 10
    if dataset == "SmoothSubspace":
        return 3
    if dataset == "UMD":
        return 3
    print("Missing dataset: %s"%dataset)
    return 2
    
def class_offset(y, dataset):
    return (y + class_modifier_add(dataset)) * class_modifier_multi(dataset)
    
def class_modifier_add(dataset):
    if dataset == "FiftyWords":
        return -1 #270
    if dataset == "Adiac":
        return -1 #176
    if dataset == "ArrowHead":
        return 0 #251
    if dataset == "Beef":
        return -1 #470
    if dataset == "BeetleFly":
        return -1 #512
    if dataset == "BirdChicken":
        return -1 #512
    if dataset == "Car":
        return -1 #577
    if dataset == "CBF":
        return -1 #128
    if dataset == "ChlorineConcentration":
        return -1 #166
    if dataset == "CinCECGTorso":
        return -1 #1639
    if dataset == "Coffee":
        return 0 #286
    if dataset == "Computers":
        return -1 #720
    if dataset == "CricketX":
        return -1 #300
    if dataset == "CricketY":
        return -1 #300
    if dataset == "CricketZ":
        return -1 #300
    if dataset == "DiatomSizeReduction":
        return -1 #345
    if dataset == "DistalPhalanxOutlineAgeGroup":
        return -1 #80
    if dataset == "DistalPhalanxOutlineCorrect":
        return 0 #80
    if dataset == "DistalPhalanxTW":
        return -3 #80
    if dataset == "Earthquakes":
        return 0 #512
    if dataset == "ECG200":
        return 1 #96
    if dataset == "ECG5000":
        return -1 #140
    if dataset == "ECGFiveDays":
        return -1 #136
    if dataset == "ElectricDevices":
        return -1 #96
    if dataset == "FaceAll":
        return -1 # 131
    if dataset == "FaceFour":
        return -1 # 350
    if dataset == "FacesUCR":
        return -1 # 131
    if dataset == "Fish":
        return -1 # 463
    if dataset == "FordA":
        return 1 #500
    if dataset == "FordB":
        return 1 # 500
    if dataset == "GunPoint":
        return -1 # 150
    if dataset == "Ham":
        return -1 # 431
    if dataset == "HandOutlines":
        return 0 # 2709
    if dataset == "Haptics":
        return -1 # 1092
    if dataset == "Herring":
        return -1 # 512
    if dataset == "InlineSkate":
        return -1 # 1882
    if dataset == "InsectWingbeatSound":
        return -1 # 256
    if dataset == "ItalyPowerDemand":
        return -1 # 24
    if dataset == "LargeKitchenAppliances":
        return -1 # 720
    if dataset == "Lightning2":
        return 1 # 637
    if dataset == "Lightning7":
        return 0 # 319
    if dataset == "Mallat":
        return -1 # 1024
    if dataset == "Meat":
        return -1 # 448
    if dataset == "MedicalImages":
        return -1 # 99
    if dataset == "MiddlePhalanxOutlineAgeGroup":
        return -1 #80
    if dataset == "MiddlePhalanxOutlineCorrect":
        return 0 #80
    if dataset == "MiddlePhalanxTW":
        return -3 #80
    if dataset == "MoteStrain":
        return -1 #84
    if dataset == "NonInvasiveFetalECGThorax1":
        return -1 #750
    if dataset == "NonInvasiveFetalECGThorax2":
        return -1 #750
    if dataset == "OliveOil":
        return -1 #570
    if dataset == "OSULeaf":
        return -1 #427
    if dataset == "PhalangesOutlinesCorrect":
        return 0 #80
    if dataset == "Phoneme":
        return -1 #1024
    if dataset == "Plane":
        return -1 #144
    if dataset == "ProximalPhalanxOutlineAgeGroup":
        return -1 #80
    if dataset == "ProximalPhalanxOutlineCorrect":
        return 0 #80
    if dataset == "ProximalPhalanxTW":
        return -3 #80
    if dataset == "RefrigerationDevices":
        return -1 #720
    if dataset == "ScreenType":
        return -1 #720
    if dataset == "ShapeletSim":
        return 0 #500
    if dataset == "ShapesAll":
        return -1 # 512
    if dataset == "SmallKitchenAppliances":
        return -1 #720
    if dataset == "SonyAIBORobotSurface2":
        return -1 #65
    if dataset == "SonyAIBORobotSurface1":
        return -1 #70
    if dataset == "StarLightCurves":
        return -1 #1024
    if dataset == "Strawberry":
        return -1 #235
    if dataset == "SwedishLeaf":
        return -1 # 128
    if dataset == "Symbols":
        return -1 #398
    if dataset == "SyntheticControl":
        return -1 #60
    if dataset == "ToeSegmentation1":
        return 0 #277
    if dataset == "ToeSegmentation2":
        return 0 #343
    if dataset == "Trace":
        return -1 #275
    if dataset == "TwoLeadECG":
        return -1 #82
    if dataset == "TwoPatterns":
        return -1 #128
    if dataset == "UWaveGestureLibraryX":
        return -1 # 315
    if dataset == "UWaveGestureLibraryY":
        return -1 # 315
    if dataset == "UWaveGestureLibraryZ":
        return -1 # 315
    if dataset == "UWaveGestureLibraryAll":
        return -1 # 945
    if dataset == "Wafer":
        return 1 #152
    if dataset == "Wine":
        return -1 #234
    if dataset == "WordsSynonyms":
        return -1 #270
    if dataset == "Worms":
        return -1 #900
    if dataset == "WormsTwoClass":
        return -1 #900
    if dataset == "Yoga":
        return -1 #426
    
    if dataset == "ACSF1":
        return 0 
    if dataset == "AllGestureWiimoteX":
        return -1 
    if dataset == "AllGestureWiimoteY":
        return -1
    if dataset == "AllGestureWiimoteZ":
        return -1
    if dataset == "BME":
        return -1
    if dataset == "Chinatown":
        return -1
    if dataset == "Crop":
        return -1
    if dataset == "DodgerLoopDay":
        return -1
    if dataset == "DodgerLoopGame":
        return -1
    if dataset == "DodgerLoopWeekend":
        return -1
    if dataset == "EOGHorizontalSignal":
        return -1
    if dataset == "EOGVerticalSignal":
        return -1
    if dataset == "EthanolLevel":
        return -1
    if dataset == "FreezerRegularTrain":
        return -1
    if dataset == "FreezerSmallTrain":
        return -1
    if dataset == "Fungi":
        return -1
    if dataset == "GestureMidAirD1":
        return -1
    if dataset == "GestureMidAirD2":
        return -1
    if dataset == "GestureMidAirD3":
        return -1
    if dataset == "GesturePebbleZ1":
        return -1
    if dataset == "GesturePebbleZ2":
        return -1
    if dataset == "GunPointAgeSpan":
        return -1
    if dataset == "GunPointMaleVersusFemale":
        return -1
    if dataset == "GunPointOldVersusYoung":
        return -1
    if dataset == "HouseTwenty":
        return -1
    if dataset == "InsectEPGRegularTrain":
        return -1
    if dataset == "InsectEPGSmallTrain":
        return -1
    if dataset == "MelbournePedestrian":
        return -1
    if dataset == "MixedShapesRegularTrain":
        return -1
    if dataset == "MixedShapesSmallTrain":
        return -1
    if dataset == "PickupGestureWiimoteZ":
        return -1
    if dataset == "PigAirwayPressure":
        return -1
    if dataset == "PigArtPressure":
        return -1
    if dataset == "PigCVP":
        return -1
    if dataset == "PLAID":
        return 0
    if dataset == "PowerCons":
        return -1
    if dataset == "Rock":
        return -1
    if dataset == "SemgHandGenderCh2":
        return -1
    if dataset == "SemgHandMovementCh2":
        return -1
    if dataset == "SemgHandSubjectCh2":
        return -1
    if dataset == "ShakeGestureWiimoteZ":
        return -1
    if dataset == "SmoothSubspace":
        return -1
    if dataset == "UMD":
        return -1
    return 0

def class_modifier_multi(dataset):
    if dataset == "ECG200":
        return 0.5 #96
    if dataset == "FordA":
        return 0.5 #500
    if dataset == "FordB":
        return 0.5 # 500
    if dataset == "Lightning2":
        return 0.5 # 637
    if dataset == "Wafer":
        return 0.5 #152
    return 1
