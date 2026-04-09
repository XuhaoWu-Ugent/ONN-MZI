# MZI V_max / V_min Extraction

Non-integer voltages from data files = precise extremum markers.
User-provided values for MZI 0,5,10,15,20,25,30,35,40,49.

Please verify: V_max = first transmission maximum, V_min = first transmission minimum (V_pi).

## MZI00 (R = 1228.6 ohm)
- Source: user-provided
- V_max = 1.857 V
- V_min = 4.491 V

## MZI01 (R = 1259.3 ohm)
- Source: data file markers
- Non-integer voltages found: [3.9, 5.8]
  - V=3.90: in2->out2: 1.86e-04 mW, in2->out3: 9.95e-06 mW
  - V=5.80: in2->out2: 5.38e-06 mW, in2->out3: 3.55e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 3.90**
- Note: high-V markers [5.8] may be 2nd-period extrema

## MZI02 (R = 1235.4 ohm)
- Source: data file markers
- Non-integer voltages found: [0.9, 4.4, 6.2]
  - V=0.90: in4->out4: 2.07e-06 mW, in4->out5: 4.09e-04 mW
  - V=4.40: in4->out4: 4.29e-04 mW, in4->out5: 6.63e-06 mW
  - V=6.20: in4->out4: 2.99e-06 mW, in4->out5: 4.62e-04 mW
- **Suggested V_max = 0.90**
- **Suggested V_min = 4.40**
- Note: high-V markers [6.2] may be 2nd-period extrema

## MZI03 (R = 1227.3 ohm)
- Source: data file markers
- Non-integer voltages found: [3.9, 5.9]
  - V=3.90: in6->out6: 3.07e-04 mW, in6->out7: 8.41e-08 mW
  - V=5.90: in6->out6: 3.43e-06 mW, in6->out7: 1.62e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 3.90**
- Note: high-V markers [5.9] may be 2nd-period extrema

## MZI04 (R = 1240.0 ohm)
- Source: data file markers
- Non-integer voltages found: [1.2, 4.3]
  - V=1.20: in8->out8: 2.71e-05 mW, in8->out9: 2.44e-04 mW
  - V=4.30: in8->out8: 3.91e-04 mW, in8->out9: 1.04e-05 mW
- **Suggested V_max = 1.20**
- **Suggested V_min = 4.30**

## MZI05 (R = 1252.5 ohm)
- Source: user-provided
- V_max = 1.477 V
- V_min = 4.682 V

## MZI06 (R = 1231.9 ohm)
- Source: data file markers
- Non-integer voltages found: [0.6, 4.4, 4.5, 6.2, 8.8]
  - V=0.60: in2->out2: 1.58e-05 mW, in2->out3: 3.28e-04 mW
  - V=4.40: in2->out2: 1.73e-07 mW, in2->out3: 9.60e-08 mW, in5->out4: 4.98e-05 mW, in5->out5: 9.78e-05 mW
  - V=4.50: in4->out4: 8.53e-05 mW, in4->out5: 1.80e-04 mW
  - V=6.20: in4->out4: 1.24e-06 mW, in4->out5: 4.48e-04 mW
  - V=8.80: in4->out4: 6.55e-06 mW, in4->out5: 4.29e-04 mW
- **Suggested V_max = 0.60**
- **Suggested V_min = 4.40**
- Note: high-V markers [6.2, 8.8] may be 2nd-period extrema

## MZI07 (R = 1258.2 ohm)
- Source: data file markers
- Non-integer voltages found: [1.4, 4.6, 6.3]
  - V=1.40: in4->out4: 1.24e-05 mW, in4->out5: 4.11e-04 mW
  - V=4.60: in4->out4: 1.29e-04 mW, in4->out5: 3.07e-05 mW
  - V=6.30: in4->out4: 1.87e-05 mW, in4->out5: 4.18e-04 mW
- **Suggested V_max = 1.40**
- **Suggested V_min = 4.60**
- Note: high-V markers [6.3] may be 2nd-period extrema

## MZI08 (R = 1268.7 ohm)
- Source: data file markers
- Non-integer voltages found: [1.4, 4.5, 4.6, 6.3]
  - V=1.40: in9->out8: 4.13e-04 mW, in9->out9: 3.01e-05 mW
  - V=4.50: in9->out8: 4.17e-07 mW, in9->out9: 2.16e-05 mW
  - V=4.60: in6->out6: 1.16e-05 mW, in6->out7: 1.45e-07 mW
  - V=6.30: in6->out6: 1.50e-05 mW, in6->out7: 1.62e-04 mW
- **Suggested V_max = 1.40**
- **Suggested V_min = 4.50**
- Note: high-V markers [6.3] may be 2nd-period extrema

## MZI09 (R = 1236.7 ohm)
- Source: data file markers
- Non-integer voltages found: [1.7, 4.5]
  - V=1.70: in1->out0: 3.36e-04 mW, in1->out1: 3.45e-05 mW
  - V=4.50: in1->out0: 1.14e-07 mW, in1->out1: 1.16e-05 mW
- **Suggested V_max = 1.70**
- **Suggested V_min = 4.50**

## MZI10 (R = 1220.8 ohm)
- Source: user-provided
- V_max = 1.202 V
- V_min = 4.208 V

## MZI11 (R = 1230.1 ohm)
- Source: data file markers
- Non-integer voltages found: [1.8, 4.5]
  - V=1.80: in2->out2: 1.37e-05 mW, in2->out3: 3.62e-04 mW
  - V=4.50: in2->out2: 2.07e-06 mW, in2->out3: 2.41e-06 mW
- **Suggested V_max = 1.80**
- **Suggested V_min = 4.50**

## MZI12 (R = 1222.2 ohm)
- Source: data file markers
- Non-integer voltages found: [7.1]
  - V=7.10: in4->out4: 1.20e-04 mW, in4->out5: 4.23e-05 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 7.10**
- Note: high-V markers [7.1] may be 2nd-period extrema

## MZI13 (R = 1240.9 ohm)
- Source: data file markers
- Non-integer voltages found: [4.3, 5.9]
  - V=4.30: in8->out8: 6.79e-05 mW, in8->out9: 6.67e-06 mW
  - V=5.90: in6->out6: 1.34e-05 mW, in6->out7: 1.51e-04 mW, in8->out8: 2.20e-05 mW, in8->out9: 2.46e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.30**
- Note: high-V markers [5.9] may be 2nd-period extrema

## MZI14 (R = 1244.9 ohm)
- Source: data file markers
- Non-integer voltages found: [4.2, 5.9]
  - V=4.20: in5->out4: 4.79e-05 mW, in5->out5: 9.72e-05 mW
  - V=5.90: in5->out4: 4.10e-04 mW, in5->out5: 1.04e-06 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.20**
- Note: high-V markers [5.9] may be 2nd-period extrema

## MZI15 (R = 1255.8 ohm)
- Source: user-provided
- V_max = 1.987 V
- V_min = 4.790 V

## MZI16 (R = 1250.0 ohm)
- Source: data file markers
- Non-integer voltages found: [1.9, 4.7]
  - V=1.90: in2->out2: 2.02e-05 mW, in2->out3: 3.74e-04 mW
  - V=4.70: in2->out2: 1.06e-06 mW, in2->out3: 4.56e-07 mW
- **Suggested V_max = 1.90**
- **Suggested V_min = 4.70**

## MZI17 (R = 1263.2 ohm)
- Source: data file markers
- Non-integer voltages found: [1.5, 2.2, 4.5, 4.9, 6.2]
  - V=1.50: in4->out4: 1.14e-05 mW, in4->out5: 4.00e-04 mW
  - V=2.20: in8->out8: 9.49e-06 mW, in8->out9: 2.94e-04 mW
  - V=4.50: in4->out4: 1.45e-04 mW, in4->out5: 2.04e-05 mW
  - V=4.90: in8->out8: 1.89e-05 mW, in8->out9: 1.55e-06 mW
  - V=6.20: in4->out4: 4.01e-05 mW, in4->out5: 3.91e-04 mW
- **Suggested V_max = 1.50**
- **Suggested V_min = 4.50**
- Note: high-V markers [6.2] may be 2nd-period extrema

## MZI18 (R = 1231.5 ohm)
- Source: data file markers
- Non-integer voltages found: [4.4]
  - V=4.40: in5->out4: 7.91e-05 mW, in5->out5: 1.41e-04 mW
- **Suggested V_max = 0.00**
- **Suggested V_min = 4.40**

## MZI19 (R = 1238.1 ohm)
- Source: data file markers
- Non-integer voltages found: [1.4, 4.4]
  - V=1.40: in7->out6: 3.17e-04 mW, in7->out7: 1.15e-05 mW
  - V=4.40: in7->out6: 7.81e-07 mW, in7->out7: 8.14e-06 mW
- **Suggested V_max = 1.40**
- **Suggested V_min = 4.40**

## MZI20 (R = 1238.3 ohm)
- Source: user-provided
- V_max = 5.780 V
- V_min = 4.179 V

## MZI21 (R = 1248.0 ohm)
- Source: data file markers
- Non-integer voltages found: [3.9, 5.7]
  - V=3.90: in2->out2: 2.19e-06 mW, in2->out3: 2.09e-07 mW
  - V=5.70: in2->out2: 2.62e-05 mW, in2->out3: 3.80e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 3.90**
- Note: high-V markers [5.7] may be 2nd-period extrema

## MZI22 (R = 1234.4 ohm)
- Source: data file markers
- Non-integer voltages found: [4.1, 4.2, 5.9, 7.2]
  - V=4.10: in4->out4: 1.16e-04 mW, in4->out5: 3.99e-05 mW
  - V=4.20: in6->out6: 1.14e-05 mW, in6->out7: 2.25e-07 mW
  - V=5.90: in6->out6: 1.50e-05 mW, in6->out7: 1.55e-04 mW
  - V=7.20: in4->out4: 1.32e-04 mW, in4->out5: 3.24e-05 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.10**
- Note: high-V markers [5.9, 7.2] may be 2nd-period extrema

## MZI23 (R = 1254.5 ohm)
- Source: data file markers
- Non-integer voltages found: [1.2, 4.4]
  - V=1.20: in7->out6: 3.10e-04 mW, in7->out7: 1.28e-05 mW
  - V=4.40: in7->out6: 7.15e-08 mW, in7->out7: 7.49e-06 mW
- **Suggested V_max = 1.20**
- **Suggested V_min = 4.40**

## MZI24 (R = 1256.4 ohm)
- Source: data file markers
- Non-integer voltages found: [4.3]
  - V=4.30: in9->out8: 1.83e-06 mW, in9->out9: 1.93e-05 mW
- **Suggested V_max = 0.00**
- **Suggested V_min = 4.30**

## MZI25 (R = 1266.7 ohm)
- Source: user-provided
- V_max = 0.000 V
- V_min = 4.077 V

## MZI26 (R = 1258.0 ohm)
- Source: data file markers
- Non-integer voltages found: [2.5, 4.9, 9.1]
  - V=2.50: in2->out2: 2.31e-05 mW, in2->out3: 4.81e-04 mW
  - V=4.90: in2->out2: 9.28e-07 mW, in2->out3: 8.50e-07 mW
  - V=9.10: in6->out6: 1.12e-05 mW, in6->out7: 2.11e-04 mW
- **Suggested V_max = 2.50**
- **Suggested V_min = 4.90**
- Note: high-V markers [9.1] may be 2nd-period extrema

## MZI27 (R = 1239.2 ohm)
- Source: data file markers
- Non-integer voltages found: [4.6]
  - V=4.60: in5->out4: 4.75e-05 mW, in5->out5: 1.05e-04 mW
- **Suggested V_max = 0.00**
- **Suggested V_min = 4.60**

## MZI28 (R = 1248.7 ohm)
- Source: data file markers
- Non-integer voltages found: [1.4, 4.3]
  - V=1.40: in9->out8: 4.12e-04 mW, in9->out9: 2.64e-05 mW
  - V=4.30: in9->out8: 4.80e-07 mW, in9->out9: 2.15e-05 mW
- **Suggested V_max = 1.40**
- **Suggested V_min = 4.30**

## MZI29 (R = 1244.8 ohm)
- Source: data file markers
- Non-integer voltages found: [4.4, 6.1]
  - V=4.40: in8->out8: 3.74e-05 mW, in8->out9: 4.10e-06 mW
  - V=6.10: in8->out8: 1.13e-05 mW, in8->out9: 2.50e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.40**
- Note: high-V markers [6.1] may be 2nd-period extrema

## MZI30 (R = 1243.9 ohm)
- Source: user-provided
- V_max = 0.000 V
- V_min = 4.186 V

## MZI31 (R = 1241.9 ohm)
- Source: data file markers
- Non-integer voltages found: [0.5, 4.1, 4.2, 5.8]
  - V=0.50: in2->out2: 1.40e-05 mW, in2->out3: 3.55e-04 mW
  - V=4.10: in4->out4: 1.16e-04 mW, in4->out5: 3.12e-05 mW
  - V=4.20: in2->out2: 4.06e-06 mW, in2->out3: 7.38e-08 mW
  - V=5.80: in4->out4: 1.43e-05 mW, in4->out5: 3.99e-04 mW
- **Suggested V_max = 0.50**
- **Suggested V_min = 4.10**
- Note: high-V markers [5.8] may be 2nd-period extrema

## MZI32 (R = 1254.5 ohm)
- Source: data file markers
- Non-integer voltages found: [4.2]
  - V=4.20: in9->out8: 6.34e-07 mW, in9->out9: 2.30e-05 mW
- **Suggested V_max = 0.00**
- **Suggested V_min = 4.20**

## MZI33 (R = 1259.2 ohm)
- Source: data file markers
- Non-integer voltages found: [5.1, 6.7]
  - V=5.10: in8->out8: 2.65e-05 mW, in8->out9: 2.45e-06 mW
  - V=6.70: in8->out8: 8.65e-06 mW, in8->out9: 3.39e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 5.10**
- Note: high-V markers [6.7] may be 2nd-period extrema

## MZI34 (R = 1262.3 ohm)
- Source: data file markers
- Non-integer voltages found: [2.6]
  - V=2.60: in6->out6: 1.50e-05 mW, in6->out7: 2.17e-04 mW
- **Suggested V_max = 2.60**
- **Suggested V_min = N/A**

## MZI35 (R = 1265.6 ohm)
- Source: user-provided
- V_max = 0.000 V
- V_min = 4.091 V

## MZI36 (R = 1247.1 ohm)
- Source: data file markers
- Non-integer voltages found: [1.5, 4.4]
  - V=1.50: in9->out8: 4.15e-04 mW, in9->out9: 3.32e-05 mW
  - V=4.40: in9->out8: 8.80e-07 mW, in9->out9: 2.77e-05 mW
- **Suggested V_max = 1.50**
- **Suggested V_min = 4.40**

## MZI37 (R = 1253.5 ohm)
- Source: data file markers
- Non-integer voltages found: [2.1, 3.8, 4.7, 5.7]
  - V=2.10: in8->out8: 8.71e-06 mW, in8->out9: 2.80e-04 mW
  - V=3.80: in6->out6: 1.85e-05 mW, in6->out7: 1.29e-04 mW
  - V=4.70: in8->out9: 1.98e-06 mW
  - V=5.70: in6->out6: 1.64e-05 mW, in6->out7: 1.57e-04 mW
- **Suggested V_max = 2.10**
- **Suggested V_min = 3.80**
- Note: high-V markers [5.7] may be 2nd-period extrema

## MZI38 (R = 1243.4 ohm)
- Source: data file markers
- Non-integer voltages found: [2.5, 4.9, 8.9]
  - V=2.50: in6->out6: 2.38e-05 mW, in6->out7: 1.95e-04 mW
  - V=4.90: in6->out6: 1.62e-05 mW, in6->out7: 3.42e-08 mW
  - V=8.90: in6->out6: 1.24e-05 mW, in6->out7: 2.01e-04 mW
- **Suggested V_max = 2.50**
- **Suggested V_min = 4.90**
- Note: high-V markers [8.9] may be 2nd-period extrema

## MZI39 (R = 1251.0 ohm)
- Source: data file markers
- Non-integer voltages found: [4.1, 5.8, 7.2]
  - V=4.10: in4->out4: 8.87e-05 mW, in4->out5: 5.51e-05 mW
  - V=5.80: in4->out4: 9.42e-06 mW, in4->out5: 4.03e-04 mW
  - V=7.20: in4->out4: 1.09e-04 mW, in4->out5: 5.26e-05 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.10**
- Note: high-V markers [5.8, 7.2] may be 2nd-period extrema

## MZI40 (R = 1247.3 ohm)
- Source: user-provided
- V_max = 0.835 V
- V_min = 4.240 V

## MZI41 (R = 1264.8 ohm)
- Source: data file markers
- Non-integer voltages found: [4.3, 6.2]
  - V=4.30: in8->out8: 3.22e-05 mW, in8->out9: 1.84e-06 mW
  - V=6.20: in8->out8: 1.86e-05 mW, in8->out9: 2.58e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.30**
- Note: high-V markers [6.2] may be 2nd-period extrema

## MZI42 (R = 1262.9 ohm)
- Source: data file markers
- Non-integer voltages found: [6.1]
  - V=6.10: in6->out6: 1.66e-05 mW, in6->out7: 1.63e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 6.10**
- Note: high-V markers [6.1] may be 2nd-period extrema

## MZI43 (R = 1265.0 ohm)
- Source: data file markers
- Non-integer voltages found: [4.2, 6.1, 8.9]
  - V=4.20: in4->out4: 1.18e-04 mW
  - V=6.10: in4->out4: 5.96e-07 mW, in4->out5: 4.51e-04 mW
  - V=8.90: in4->out4: 1.06e-05 mW, in4->out5: 4.52e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.20**
- Note: high-V markers [6.1, 8.9] may be 2nd-period extrema

## MZI44 (R = 1279.9 ohm)
- Source: data file markers
- Non-integer voltages found: [4.4]
  - V=4.40: in2->out2: 2.26e-05 mW, in2->out3: 1.74e-06 mW
- **Suggested V_max = 0.00**
- **Suggested V_min = 4.40**

## MZI45 (R = 1208.3 ohm)
- Source: data file markers
- Non-integer voltages found: [1.4, 4.4, 6.1]
  - V=1.40: in8->out8: 1.57e-05 mW, in8->out9: 2.47e-04 mW, in9->out8: 4.13e-04 mW, in9->out9: 7.71e-06 mW
  - V=4.40: in8->out8: 2.71e-04 mW, in8->out9: 1.73e-05 mW, in9->out8: 1.04e-05 mW, in9->out9: 3.85e-04 mW
  - V=6.10: in8->out8: 1.45e-05 mW, in8->out9: 2.47e-04 mW
- **Suggested V_max = 1.40**
- **Suggested V_min = 4.40**
- Note: high-V markers [6.1] may be 2nd-period extrema

## MZI46 (R = 1250.7 ohm)
- Source: data file markers
- Non-integer voltages found: [4.6, 6.4]
  - V=4.60: in6->out6: 1.97e-04 mW, in6->out7: 8.30e-06 mW
  - V=6.40: in6->out6: 1.28e-05 mW, in6->out7: 1.54e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.60**
- Note: high-V markers [6.4] may be 2nd-period extrema

## MZI47 (R = 1249.5 ohm)
- Source: data file markers
- Non-integer voltages found: [4.3, 6.1]
  - V=4.30: in4->out4: 4.72e-04 mW, in4->out5: 4.20e-07 mW
  - V=6.10: in4->out4: 2.78e-06 mW, in4->out5: 4.36e-04 mW
- **Suggested V_max = N/A**
- **Suggested V_min = 4.30**
- Note: high-V markers [6.1] may be 2nd-period extrema

## MZI48 (R = 1226.3 ohm)
- Source: data file markers
- Non-integer voltages found: [1.5, 4.6]
  - V=1.50: in2->out2: 2.78e-07 mW, in2->out3: 3.66e-04 mW
  - V=4.60: in2->out2: 3.59e-04 mW, in2->out3: 7.34e-07 mW
- **Suggested V_max = 1.50**
- **Suggested V_min = 4.60**

## MZI49 (R = 1230.9 ohm)
- Source: user-provided
- V_max = 0.000 V
- V_min = 4.286 V
