import numpy as np
import pandas as pd
import torch
import json
import os
from scipy.interpolate import interp1d
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RawSinThetaToVoltageConverter:
    """
    Complete conversion pipeline: raw_sin_theta -> sin_theta -> phase -> voltage
    
    Conversion chain:
    1. raw_sin_theta (from hook data) -> sin_theta (via sigmoid transform)
    2. sin_theta -> phase_theta (via arcsin) 
    3. phase_theta -> voltage (via lookup table interpolation)
    """
    
    def __init__(self, voltage_phase_csv_path="Voltage_Phase_Correspondence.csv"):
        """
        Initialize converter with voltage-phase lookup table
        
        Args:
            voltage_phase_csv_path: Path to voltage-phase correspondence CSV file
        """
        self.voltage_phase_csv_path = voltage_phase_csv_path
        self.voltage_to_phase_interp = None
        self.phase_to_voltage_interp = None
        self.voltage_range = None
        self.phase_range = None
        
        self._load_voltage_phase_lookup_table()
    
    def _load_voltage_phase_lookup_table(self):
        """
        Load voltage-phase data and extract physical relationship coefficients
        Uses the physical relationship: phi = k * V^2 + offset
        """
        try:
            df = pd.read_csv(self.voltage_phase_csv_path)
            
            voltage_data = df['Voltage_V'].values
            phase_data = df['Phase_Difference_rad'].values
            
            self.voltage_range = (voltage_data.min(), voltage_data.max())
            self.phase_range = (phase_data.min(), phase_data.max())
            
            # Extract physical relationship coefficients: phi = k * V^2 + offset
            voltage_squared = voltage_data ** 2
            coeffs = np.polyfit(voltage_squared, phase_data, 1)
            self.k_coefficient = coeffs[0]  # Phase scale coefficient (rad/V^2)
            self.offset = coeffs[1]         # Phase offset (rad)
            
            # Calculate fit quality
            phase_predicted = self.k_coefficient * voltage_squared + self.offset
            ss_res = np.sum((phase_data - phase_predicted) ** 2)
            ss_tot = np.sum((phase_data - np.mean(phase_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Voltage-Phase physical relationship extracted successfully")
            print(f"Physical formula: phi = {self.k_coefficient:.8f} * V^2 + {self.offset:.8f}")
            print(f"Inverse formula: V = sqrt((phi - {self.offset:.8f}) / {self.k_coefficient:.8f})")
            print(f"Fit quality: R^2 = {r_squared:.6f}")
            print(f"Voltage range: {self.voltage_range[0]:.3f}V to {self.voltage_range[1]:.3f}V")
            print(f"Phase range: {self.phase_range[0]:.3f} to {self.phase_range[1]:.3f} rad")
            print(f"Phase range: {self.phase_range[0]*180/np.pi:.1f} to {self.phase_range[1]*180/np.pi:.1f} deg")
            
        except Exception as e:
            print(f"Error loading voltage-phase lookup table: {e}")
            raise
    
    def raw_sin_theta_to_sin_theta(self, raw_sin_theta):
        """
        Convert raw_sin_theta to sin_theta using sigmoid transformation
        
        Args:
            raw_sin_theta: Raw parameter from MZI (tensor or float/array)
            
        Returns:
            sin_theta: Transformed sine value [-1, 1]
        """
        if torch.is_tensor(raw_sin_theta):
            raw_sin_theta = raw_sin_theta.detach().cpu().numpy()
        
        # Apply sigmoid transformation: sin_theta = 2 * sigmoid(raw_sin_theta) - 1
        sin_theta = 2 * (1 / (1 + np.exp(-raw_sin_theta))) - 1
        
        return sin_theta
    
    def sin_theta_to_phase(self, sin_theta):
        """
        Convert sin_theta to phase angle
        
        Args:
            sin_theta: Sine value [-1, 1]
            
        Returns:
            phase_theta: Phase angle in radians
        """
        # Use arcsin to get phase angle
        # Note: arcsin gives values in [-π/2, π/2]
        phase_theta = np.arcsin(np.clip(sin_theta, -1, 1))
        
        return phase_theta
    
    def phase_to_voltage(self, phase_theta):
        """
        Convert phase angle to control voltage using physical relationship
        Uses: V = sqrt((phi - offset) / k_coefficient)
        
        Args:
            phase_theta: Phase angle in radians
            
        Returns:
            voltage: Control voltage in Volts
        """
        # Handle scalar and array inputs
        is_scalar = np.isscalar(phase_theta)
        if is_scalar:
            phase_theta = np.array([phase_theta])
        
        # Handle negative phases using 2*pi periodicity: phi = phi + 2*pi
        phase_positive = np.where(phase_theta < 0, phase_theta + 2*np.pi, phase_theta)
        
        # Use physical formula: V = sqrt((phi - offset) / k_coefficient)
        voltage = np.sqrt(np.maximum(0, (phase_positive - self.offset) / self.k_coefficient))
        
        # Clip to valid voltage range
        voltage = np.clip(voltage, self.voltage_range[0], self.voltage_range[1])
        
        if is_scalar:
            return voltage[0]
        return voltage
    
    def convert_single_raw_sin_theta(self, raw_sin_theta, verbose=False):
        """
        Complete conversion pipeline for a single raw_sin_theta value
        
        Args:
            raw_sin_theta: Raw parameter value
            verbose: Print intermediate steps
            
        Returns:
            dict: Contains all intermediate and final values
        """
        # Step 1: raw_sin_theta -> sin_theta
        sin_theta = self.raw_sin_theta_to_sin_theta(raw_sin_theta)
        
        # Step 2: sin_theta -> phase_theta  
        phase_theta = self.sin_theta_to_phase(sin_theta)
        
        # Step 3: phase_theta -> voltage
        voltage = self.phase_to_voltage(phase_theta)
        
        result = {
            'raw_sin_theta': raw_sin_theta,
            'sin_theta': sin_theta,
            'cos_theta': np.sqrt(1 - sin_theta**2),  # For completeness
            'phase_theta_rad': phase_theta,
            'phase_theta_deg': phase_theta * 180 / np.pi,
            'control_voltage_V': voltage
        }
        
        if verbose:
            print(f"Conversion results:")
            print(f"  raw_sin_theta: {raw_sin_theta:.6f}")
            print(f"  sin_theta: {sin_theta:.6f}")
            print(f"  cos_theta: {result['cos_theta']:.6f}")
            print(f"  phase_theta: {phase_theta:.6f} rad = {result['phase_theta_deg']:.2f} deg")
            print(f"  control_voltage: {voltage:.6f} V")
        
        return result
    
    def convert_hook_data(self, hook_data_path, output_path=None):
        """
        Convert hook data containing raw_sin_theta values to control voltages
        
        Args:
            hook_data_path: Path to numpy file containing hook data
            output_path: Path to save conversion results (optional)
            
        Returns:
            dict: Converted data structure with voltage values
        """
        try:
            # Load hook data
            hook_data = np.load(hook_data_path, allow_pickle=True).item()
            print(f"Hook data loaded from: {hook_data_path}")
            
            # Create results structure
            conversion_results = {
                'input_image_to_model': hook_data.get('input_image_to_model'),
                'final_model_output': hook_data.get('final_model_output'),
                'filter_conversions': {}
            }
            
            filter_internals = hook_data.get('filter_internals', {})
            
            for filter_key, filter_data in filter_internals.items():
                print(f"\nProcessing {filter_key}...")
                
                filter_conversion = {
                    'mzi_array_input': filter_data.get('mzi_array_input'),
                    'mzi_output_before_absorber': filter_data.get('mzi_output_before_absorber'),
                    'output_after_absorber': filter_data.get('output_after_absorber'),
                    'mzi_array_io_waveguide_description': filter_data.get('mzi_array_io_waveguide_description'),
                    'mzi_voltage_settings': []
                }
                
                raw_thetas_data = filter_data.get('raw_sin_thetas_with_position', [])
                
                for i, theta_data in enumerate(raw_thetas_data):
                    raw_sin_theta = theta_data['value']
                    position = theta_data['position']
                    gradient = theta_data.get('gradient')
                    
                    # Convert raw_sin_theta to voltage
                    conversion = self.convert_single_raw_sin_theta(raw_sin_theta)
                    
                    # Create MZI setting record
                    mzi_setting = {
                        'mzi_position': position,
                        'raw_sin_theta': raw_sin_theta,
                        'gradient': gradient,
                        'conversion_results': conversion,
                        'control_voltage_V': conversion['control_voltage_V'],
                        'mzi_description': self._generate_mzi_description(position)
                    }
                    
                    filter_conversion['mzi_voltage_settings'].append(mzi_setting)
                
                conversion_results['filter_conversions'][filter_key] = filter_conversion
                print(f"  Converted {len(raw_thetas_data)} MZI settings")
            
            # Save results if output path provided
            if output_path:
                np.save(output_path, conversion_results)
                print(f"\nConversion results saved to: {output_path}")
            
            # Generate summary
            self._print_conversion_summary(conversion_results)
            
            return conversion_results
            
        except Exception as e:
            print(f"Error processing hook data: {e}")
            raise
    
    def _generate_mzi_description(self, position):
        """
        Generate human-readable MZI description
        """
        mzi_type = position.get('mzi_array_layer_type', 'unknown')
        layer_idx = position.get('scf_mzi_array_layer_index', -1)
        mzi_idx = position.get('mzi_index_within_mzi_array_layer', -1)
        waveguides_desc = position.get('mzi_acting_on_waveguides_description', '')
        
        return f"{mzi_type} Layer {layer_idx}, MZI {mzi_idx}: {waveguides_desc}"
    
    def _print_conversion_summary(self, conversion_results):
        """
        Print summary of conversion results
        """
        print(f"\n" + "="*80)
        print(f"CONVERSION SUMMARY")
        print(f"="*80)
        
        filter_conversions = conversion_results.get('filter_conversions', {})
        total_mzis = 0
        voltage_stats = []
        
        for filter_key, filter_data in filter_conversions.items():
            mzi_settings = filter_data.get('mzi_voltage_settings', [])
            num_mzis = len(mzi_settings)
            total_mzis += num_mzis
            
            if num_mzis > 0:
                voltages = [setting['control_voltage_V'] for setting in mzi_settings]
                voltage_stats.extend(voltages)
                
                print(f"\n{filter_key}:")
                print(f"  MZIs: {num_mzis}")
                print(f"  Voltage range: {np.min(voltages):.6f}V to {np.max(voltages):.6f}V")
                print(f"  Average voltage: {np.mean(voltages):.6f}V")
        
        if voltage_stats:
            print(f"\nOVERALL STATISTICS:")
            print(f"  Total MZIs processed: {total_mzis}")
            print(f"  Overall voltage range: {np.min(voltage_stats):.6f}V to {np.max(voltage_stats):.6f}V")
            print(f"  Overall average voltage: {np.mean(voltage_stats):.6f}V")
            print(f"  Voltage standard deviation: {np.std(voltage_stats):.6f}V")
    
    def save_voltage_settings_for_hardware(self, conversion_results, output_file="mzi_voltage_settings.json"):
        """
        Save voltage settings in a format suitable for hardware control
        
        Args:
            conversion_results: Results from convert_hook_data()
            output_file: Output filename for hardware settings
        """
        hardware_settings = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'conversion_info': {
                'voltage_range': self.voltage_range,
                'phase_range': self.phase_range,
                'total_mzis': 0
            },
            'filter_settings': {}
        }
        
        filter_conversions = conversion_results.get('filter_conversions', {})
        
        for filter_key, filter_data in filter_conversions.items():
            mzi_settings = filter_data.get('mzi_voltage_settings', [])
            
            filter_settings = {
                'filter_id': filter_key,
                'num_mzis': len(mzi_settings),
                'mzi_voltages': []
            }
            
            for setting in mzi_settings:
                mzi_voltage = {
                    'mzi_id': setting['mzi_description'],
                    'position': setting['mzi_position'],
                    'control_voltage_V': float(setting['control_voltage_V']),
                    'raw_sin_theta': float(setting['raw_sin_theta']),
                    'sin_theta': float(setting['conversion_results']['sin_theta']),
                    'phase_deg': float(setting['conversion_results']['phase_theta_deg'])
                }
                filter_settings['mzi_voltages'].append(mzi_voltage)
            
            hardware_settings['filter_settings'][filter_key] = filter_settings
            hardware_settings['conversion_info']['total_mzis'] += len(mzi_settings)
        
        # Save as JSON for hardware interface
        with open(output_file, 'w') as f:
            json.dump(hardware_settings, f, indent=2)
        
        print(f"\nHardware voltage settings saved to: {output_file}")
        return hardware_settings


def test_converter():
    """
    Test the converter with sample data
    """
    print("="*80)
    print("TESTING RAW_SIN_THETA TO VOLTAGE CONVERTER")
    print("="*80)
    
    # Initialize converter
    converter = RawSinThetaToVoltageConverter()
    
    # Test with sample raw_sin_theta values
    test_values = [-0.5, -0.2, 0.0, 0.2, 0.5]
    
    print(f"\nTesting with sample raw_sin_theta values:")
    print(f"-" * 60)
    
    for raw_val in test_values:
        print(f"\nraw_sin_theta = {raw_val:.3f}:")
        result = converter.convert_single_raw_sin_theta(raw_val, verbose=True)


if __name__ == "__main__":
    # Run test
    test_converter()
    
    # Example usage with hook data
    print(f"\n" + "="*80)
    print(f"EXAMPLE USAGE WITH HOOK DATA")  
    print(f"="*80)
    
    hook_file = "filter_data/hook_data_verified.npy"
    if os.path.exists(hook_file):
        converter = RawSinThetaToVoltageConverter()
        results = converter.convert_hook_data(hook_file, "voltage_conversion_results.npy")
        hardware_settings = converter.save_voltage_settings_for_hardware(results)
    else:
        print(f"Hook data file not found: {hook_file}")
        print(f"Please run main.py first to generate hook data.")