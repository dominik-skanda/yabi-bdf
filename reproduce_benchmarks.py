import subprocess
import re
import matplotlib

matplotlib.use('Agg')  # headless backend so Quick Verification also works on CI / no-display machines
import matplotlib.pyplot as plt

def run_tolerance_test():
    print("====================================================")
    print("1. RUNNING TOLERANCE AND ORDER ADHERENCE TEST (11)")
    print("====================================================")
    
    try:
        result = subprocess.run(['build/tests/yabi_test_11_tolerance_and_order'], capture_output=True, text=True, check=True)
        output = result.stdout
    except Exception as e:
        print(f"Error running system test 11: {e}")
        return

    rtols = []
    errors = []
    in_tolerance_table = False

    # Parse console output lines
    for line in output.split('\n'):
        line_stripped = line.strip()
        
        # Isolate the data table columns inside Section A
        if "rtol" in line_stripped and "rel error" in line_stripped:
            in_tolerance_table = True
            continue
        if "improved the error" in line_stripped or "B. first-order" in line_stripped:
            in_tolerance_table = False
            
        if in_tolerance_table and line_stripped:
            match = re.match(r'^([0-9eE.+-]+)\s+([0-9eE.+-]+)', line_stripped)
            if match:
                try:
                    rtols.append(float(match.group(1)))
                    errors.append(float(match.group(2)))
                except ValueError:
                    continue

    # Plot results if parsing succeeded
    if rtols and errors:
        sorted_pairs = sorted(zip(rtols, errors))
        rtols_sorted, errors_sorted = zip(*sorted_pairs)

        plt.figure(figsize=(8, 6))
        plt.loglog(rtols_sorted, errors_sorted, 'o-', label='YABI Observed Error', color='#0066cc', linewidth=2)
        plt.loglog(rtols_sorted, rtols_sorted, '--', label='Ideal Slope ($y=x$)', color='gray', alpha=0.7)
        
        plt.xlabel('Requested Relative Tolerance (RTOL)', fontsize=12)
        plt.ylabel('Observed Global Relative Error', fontsize=12)
        plt.title('YABI Work-Precision Proof: Tolerance Adherence', fontsize=14, fontweight='bold')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=11)
        
        plt.savefig('work_precision_diagram.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Generated work_precision_diagram.png successfully.")
    else:
        print("[ERROR] Could not parse RTOL/Error table fields from output.")

    # Also display the first-order slope convergence metrics on screen
    print("\n--- Summary of Step Convergence (Section B) ---")
    slopes = re.findall(r'observed slope.*', output, re.IGNORECASE)
    for slope in slopes:
        print(f"  {slope}")


def run_second_order_sensitivities():
    print("\n====================================================")
    print("2. RUNNING SECOND-ORDER SENSITIVITY CHECK (28)")
    print("====================================================")
    
    try:
        result = subprocess.run(['build/tests/yabi_test_28_second_order_sensitivity'], capture_output=True, text=True, check=True)
        print("--- Second-Order Sensitivity Terminal Output ---")
        print(result.stdout.strip())
        print("-------------------------------------------------")
    except Exception as e:
        print(f"[ERROR] Binary execution failed: {e}")


if __name__ == "__main__":
    run_tolerance_test()
    run_second_order_sensitivities()

