import sys
from functools import reduce
from itertools import product
from enum import Enum

class Junction(Enum):
    SPLIT = 1
    RESET_TRUE = 2
    RESET_FALSE = 3
    SPLIT_TOP = 4
    PASS = 5

#------------------------------------------------------------------------------------------------------#
# This function reads dimacs files & creates the clauses by it
def read_dimacs(filename):
    clauses = []
    num_vars = num_clauses = 0

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('c') or line == '':
                # Skip comments and empty lines
                continue
            elif line.startswith('p'):
                # Problem definition line
                parts = line.split()
                if parts[1] == 'cnf':
                    num_vars = int(parts[2])
                    num_clauses = int(parts[3])
            else:
                # Clause line
                clause = list(map(int, line.split()))
                if clause[-1] == 0:
                    clause = clause[:-1]  # Remove the trailing 0
                clauses.append(clause)

    return num_vars, num_clauses, clauses
#------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------#
# This function checks all the cluases and creats the binary representation of each variable.
def check_binary_x(num_vars, clauses, x_binary_false, x_binary_true):
    for i in range(num_vars):
        x_binary_false[i] = 0
        x_binary_true[i] = 0

        for clause in clauses:
            if clause[i] < 0:
                x_binary_false[i] = x_binary_false[i] << 1
                x_binary_false[i] += 1

                x_binary_true[i] = x_binary_true[i] << 1

            else:
                x_binary_true[i] = x_binary_true[i] << 1
                x_binary_true[i] += 1

                x_binary_false[i] = x_binary_false[i] << 1
#------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------#
# This function generates and prints all of the truth combinations we can have of the variables.
def generate_truth_table(num_vars):
    # Generate all combinations of truth values for the given number of variables
    truth_combinations = list(product([True, False], repeat=num_vars))

    # Create and print the header row for check
    header = "\n"
    for i in range(num_vars):
        header += f"x{i+1}\t"
    print(header)
    print("=" * len(header)*3)

    # Print each row of the truth table
    for combination in truth_combinations:
        row = ""
        for val in combination:
            if val:
            # Add "true" if the value is True
                row += "true\t"
            else:
            # Add "false" if the value is False
                row += "false\t"
        print(row)

    return truth_combinations
#------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------#
# This function checks the bitwise or of all variables for each truth combination and sets the output
def replace_truth_with_binary_and_compute_or(truth_combinations, x_binary_true, x_binary_false):
    binary_rows = []
    or_results = []
    outputs = []

    # Compute the number that all bits are 1 - biggest in the representation
    true_output_value = (2 ** num_vars) - 1

    for combination in truth_combinations:
        binary_row = []
        for i, value in enumerate(combination):
            if value:
                # Use binary value for True
                binary_row.append(x_binary_true[i])
            else:
                # Use binary value for False
                binary_row.append(x_binary_false[i])

        # Compute the bitwise OR for the row
        row_or_result = reduce(lambda x, y: x | y, binary_row)
        or_results.append(row_or_result)

        # Choose output type
        output = row_or_result == true_output_value
        outputs.append(output)

        binary_rows.append(binary_row)
    return binary_rows, or_results, outputs
#------------------------------------------------------------------------------------------------------#


# This function calculates the initial true and false blocks for each variable and stores the junctions in true_block_arr and false_block_arr
def init_variable_block():
    # for each variable compute true and false
    for var in range(num_vars):
        block_true_rows = x_binary_true[var]
        block_false_rows = x_binary_false[var]
        block_cols = (2 ** num_vars) - 1

        for col in range(block_cols+1):

            # true block
            if (col & x_binary_true[var]) == 0:
                true_block_arr[var].append([0, col, Junction.RESET_TRUE])

            elif col == x_binary_true[var]:
                true_block_arr[var].append([0, col, Junction.RESET_FALSE])

            else:
                true_block_arr[var].append([0, col, Junction.RESET_FALSE])
                if col & x_binary_true[var] != block_true_rows:
                    true_block_arr[var].append([(col & x_binary_true[var]), col, Junction.RESET_TRUE])

            # false block
            if (col & x_binary_false[var]) == 0:
                false_block_arr[var].append([0, col, Junction.RESET_TRUE])

            elif col == x_binary_false[var]:
                false_block_arr[var].append([0, col, Junction.RESET_FALSE])

            else:
                false_block_arr[var].append([0, col, Junction.RESET_FALSE])
                if col & x_binary_false[var] != block_false_rows:
                    false_block_arr[var].append([(col & x_binary_false[var]), col, Junction.RESET_TRUE])

            true_block_arr[var].append([block_true_rows, col, Junction.RESET_FALSE])
            false_block_arr[var].append([block_false_rows, col, Junction.RESET_FALSE])


#-------------------------------------------- Main ----------------------------------------------------#
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dimacs file required")
        sys.exit()

    # Dimacs file to load
    filename = sys.argv[1]
    num_vars, num_clauses, clauses = read_dimacs(filename)

    # ----Print for verification----
    print(f"Number of variables: {num_vars}")
    print(f"Number of clauses: {num_clauses}")
    print("Clauses:")
    for clause in clauses:
        print(clause)

    # ----Pre-allocate binary variables lists----
    x_binary_false = [0] * (num_vars)
    x_binary_true = [0] * (num_vars)
    check_binary_x (num_vars, clauses, x_binary_false, x_binary_true)

    # ---Print binary representations for True----
    print("\nBinary X (True):")
    for i, x in enumerate(x_binary_true):
        print(f"Variable x{i + 1} (True): {x} (Binary: {bin(x)})\t")

    # ----Print binary representations for False----
    print("\nBinary X (False):")
    for i, x in enumerate(x_binary_false):
        print(f"Variable x{i + 1} (False): {x} (Binary: {bin(x)})\t")

    # Calculate truth table and output
    truth_combinations = generate_truth_table(num_vars)
    binary_rows, or_results, output = replace_truth_with_binary_and_compute_or(truth_combinations, x_binary_true, x_binary_false)

    # ----Print the output table----
    # Create and print the header row for check
    header = "\n"
    for i in range(num_vars):
        header += f"x{i+1}\t"
    print(header)

    print("=" * len(header)*3)

    for row, or_result, output in zip(binary_rows, or_results, output):
        binary_strings = [bin(value)[2:] for value in row]
        print("\t".join(binary_strings) +
            f"\t=> Bitwise OR: {bin(or_result)[2:]} ({or_result})\tOutput: {'True' if output else 'False'}")


    true_block_arr = [[] for _ in range(num_vars)]
    false_block_arr = [[] for _ in range(num_vars)]
    init_variable_block()
    i = 1
    for b in true_block_arr:
        print("\nx",i, " true block junctions: ")
        i+=1
        for v in b:
            print(v)

    i = 1
    for b in false_block_arr:
        print("\nx",i, " false block junctions: ")
        i+=1
        for v in b:
            print(v)