# TODO List

This code is used to test the fabrication errors between the theoretical values of MZI arrays and the actual values after tape-out. The project is located in the MZI-ONN/ONN-MZI folder (i.e., the current folder).

## Preliminary

For a single MZI (Mach–Zehnder interferometer), it has two input ports and two output ports. In our experiment, only one input port is used, and we only care about the primary single output port related to this input port. Additionally, only one MZI's voltage is adjusted at a time. An MZI has an upper voltage and a lower voltage (i.e., V1, V2). In our experiment, only the upper voltage is used, while the lower voltage remains constant at zero. φ is the phase difference of light passing through the two arms of the MZI. When both upper and lower voltages are 0, φ₀ - the phase difference of the MZI without voltage - is random. There are two MMIs in the MZI, with a pair of heaters in between. The normal splitting ratio of MMI is 1:1, but due to fabrication errors, their splitting ratios are a:(1-a) and b:(1-b) respectively. Both a and b are between 0.45-0.55.

Each MZI has a certain power loss at input. For a single MZI, it can be approximated that for every 1mW of input light intensity, there is an immediate loss of L_il=0.06mW when entering the MZI. The resistances of the two heaters are R1 and R2 respectively. Their theoretical value is 1200 ohms, but the fabrication error may be δ∈(-50, 50) ohms. Since we only have one upper voltage V, R2 can be disregarded. The relationship between the phase difference induced by the two heaters and power is that every P_π=12 mW can produce a phase difference of π between the upper and lower arms. This value is fixed.

In summary, the relationship between input and output optical power in our experiment is as follows:
First output port: P_out1 = (P_in/L_il) · [ab + (1-a)(1-b) + 2√(ab(1-a)(1-b)) · sin(Δφ)]

Second port: P_out2 = (P_in/L_il) · [a(1-b) + (1-a)b - 2√(ab(1-a)(1-b))·sin(Δφ)]

Where Δφ = φ_upper - φ_lower = φ₀ + π/P_π × [V1²/(R1(1+δ₁)) - V2²/(R2(1+δ₂))]. Since V2 is 0, Δφ= φ₀ + π/P_π × V1²/(R1(1+δ₁)).



### MZI Array Structure
In this experiment, we did not actually create a complete network similar to CNN or ONN under the module. For calibration purposes, only a single channel structure similar to module/channel.py was created. The overall structure is consistent with the six rows and five columns in channel. The MZI array input is on the left and output is on the right. In rows, all 5 MZIs are placed horizontally. In columns, there are only 4 MZIs, placed vertically. Their connection logic is written as in module/channel.py. The top/bottom outputs of each row are directly connected to the next layer's row without entering the column. The column actually only processes the middle 1-8 outputs. Since the MZIs in columns are placed vertically, their upper-left input will be output to the lower-left/lower-right, and only the lower-right is passed to the next layer. The same applies to the lower-left output. Due to the incoherence of light, left-side output light does not need to be considered. If the output formula for upper-left input light is P_out2 = (P_in/L_il) · [a(1-b) + (1-a)b - 2√(ab(1-a)(1-b))·sin(Δφ)], then for the lower-left input light, its phase difference should be opposite. Also, its position for 'a' in the formula is the original 'b', and vice versa for 'b'. MZI numbering prioritizes top to bottom, then left to right, similar to:
0       9
    5
1       10
    6
2       11
    7
3       12
    8
4       13 ......

### Given Input and Actual Output Optical Power

All input data is in the MZI-ONN/ONN-MZI/data folder. Named MZI_array_output_in(0~9).txt. In each file, the first column is ## input_port input_power input_mzi_number input_mzi_voltage output_port output_power, and each subsequent line contains data like 1 5.0e-03 13 0.0 8 2.19e-05. input_mzi_voltage is the V1 of the specified MZI. output_port gives the primary output port. When calculating loss, only consider the given output port.

### Parameters to be Determined

To calibrate the errors, we need to know a, b, δ₁, and φ₀ for all MZIs. You should set them as trainable parameters of the model.

## Step 1: Reconstruct all MZI, MZI_row, and MZI_column code under MZI_array

In the original code, the only trainable parameter for MZI is raw_sin_theta, which is the phase difference between two optical paths. The original code uses this to construct the transfer matrix. In the new version, since we need to calibrate errors, we need to use the formulas mentioned in the preliminary section above to calculate the output optical power. You need to set all the parameters mentioned above as model parameters and modify the MZI, MZI_row, and MZI_column code to adapt to the new optical power calculation method.

## Step 2: Write a new version of the train file

The original old file uses the mnist dataset. In the new version, we use the actual input-output from on-chip testing to detect fabrication errors. The format is shown in MZI-ONN/ONN-MZI/data/MZI_array_output_in9.txt. Please design the reading code according to the data format. Each data file should be used for training. The input numbering is consistent with the input numbering mentioned above. Except for the specified input_mzi_number, all other MZI voltages are 0, meaning their phase difference only equals the initial phase difference. Finally, output all the required parameters for all MZIs to the result file.

## Step 3: Final Check

Please review the above requirements again and recheck the code.