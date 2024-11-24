from collections import Counter
from operator import itemgetter
from copy import copy


class CSP:
    def __init__(self, initBoard):
        # Initialize the CSP with the given board
        self.board = {}
        self.domain = {}
        self.constraints = []
        self.rows = "ABCDEFGHI"  # Rows of the Sudoku board (A-I)
        self.columns = "123456789"  # Columns of the Sudoku board (1-9)

        # Initialize the board, domain, and constraints
        for i, c in enumerate(self.rows):
            for j, d in enumerate(self.columns):
                key = c + d  # Construct the variable name (e.g., 'A1')
                offset = i * 8  # Offset for accessing the initial board
                value = int(initBoard[i + j + offset])  # Get the value from initBoard

                # Add to Board dictionary
                self.board[key] = value

                # Add to Domain dictionary: if the cell is empty (0), all values 1-9 are possible
                if value == 0:
                    self.domain[key] = [i + 1 for i in range(9)]
                else:
                    self.domain[key] = [value]

        # Append constraints for rows, columns, and 3x3 subgrids
        for i in self.rows:
            self.constraints.append([i + j for j in self.columns])  # Row constraints

        for i in self.columns:
            self.constraints.append([j + i for j in self.rows])  # Column constraints

        # 3x3 subgrid constraints
        a = [[[i + j for j in self.columns[0 + k:3 + k]] for i in self.rows] for k in range(0, 9, 3)]
        for i in range(3):
            for k in range(0, 9, 3):
                self.constraints.append([item for sublist in a[i][0 + k:3 + k] for item in sublist])


# Function to print the current state of the Sudoku board in a readable format
def print_board(csp):
    rows = "ABCDEFGHI"
    cols = "123456789"
    for i, r in enumerate(rows):
        if i in [3, 6]:
            print('------+-------+------')
        for j, c in enumerate(cols):
            if j in [3, 6]:
                print('|', end=" ")
            print(csp.board[r + c], end=" ")
        print()


# Function to print the solved Sudoku domain or the unsolved one
def print_solved_domain(csp):
    # If every domain has only one value, the Sudoku is solved
    if max([len(val) for val in csp.domain.values()]) == 1:
        solution_string = "".join([str(item) for sublist in list(csp.domain.values()) for item in sublist])
        print("Sudoku is Solved")
        print()
    else:
        print("Sudoku is Not Solved")
        print()
        print("Domain: ")
        print(csp.domain)


# Function to create all arcs from the constraints
def CreateArcs(constraints):
    arcs_queue = []
    for constraint in constraints:
        for x in constraint:
            for y in constraint:
                if x != y:  # Avoid arcs where a variable is compared to itself
                    arcs_queue.append((x, y))
    return arcs_queue


# Function to revise the domain of variable Xi based on the value of Xj
def Revise(csp, Xi, Xj):
    revised = False
    for Di in csp.domain[Xi]:
        satisfied = False
        for Dj in csp.domain[Xj]:
            if Di != Dj:  # If Di and Dj are different, the constraint is satisfied
                satisfied = True
        if not satisfied:  # If no valid value for Di, remove it from the domain
            csp.domain[Xi].remove(Di)
            revised = True
    return revised


# Function to enforce arc consistency using the AC3 algorithm
def AC3(csp):
    arcs_queue = CreateArcs(csp.constraints)  # Create all arcs
    fringe = arcs_queue.copy()  # Set up the queue of arcs to be processed
    while fringe:
        (Xi, Xj) = fringe.pop(0)  # Pop an arc
        if Revise(csp, Xi, Xj):  # If the domain of Xi is revised
            if len(csp.domain[Xi]) == 0:  # If Xi has no valid values left, return failure
                return False
            # Add all arcs that are now affected by the revision
            re_arcs = [arc for arc in arcs_queue if arc[1] == Xi and arc[0] != Xj]
            fringe.extend(re_arcs)  # Extend the fringe with the new arcs
    return True


# Function to select the next unassigned variable using the Minimum Remaining Values heuristic
def SelectUnassignVariable(assignments, csp):
    return min([item for item in csp.domain if item not in assignments.keys() and csp.board[item] == 0],
               key=lambda x: len(csp.domain[x]))


# Function to check if the current assignment is complete
def CheckAssignment(assignments, csp):
    emptyValues = [square for square in csp.board if csp.board[square] == 0]
    if not assignments:
        return False
    for val in assignments.keys():
        if len([assignments[val]]) != 1:
            return False
    for square in emptyValues:
        if square not in assignments.keys():
            return False
    return True


# Function to check if an assignment is consistent with the constraints
def CheckConsistent(assignment, csp):
    varis = [key for key in assignment.keys()]
    for var in varis:
        for constraint in csp.constraints:
            if var in constraint:
                peers = [item for item in constraint if item != var and item in assignment.keys()]
                for peer in peers:
                    if assignment[peer] == assignment[var]:
                        return False  # Assignment violates constraints (same value for two variables)
    return True


# Backtracking Search function that finds a solution using recursive backtracking
def BTS(csp):
    answer, assignment = BackTrack({}, csp)
    return assignment


# Backtracking function that tries to assign values and checks consistency
def BackTrack(assignments, csp):
    if CheckAssignment(assignments, csp):  # If the assignment is complete and valid
        return True, assignments
    var = SelectUnassignVariable(assignments, csp)  # Choose the next unassigned variable
    for value in csp.domain[var]:  # Try every value in the domain of the variable
        test = assignments.copy()
        test[var] = value
        consistent = CheckConsistent(test, csp)
        if consistent:
            result = BackTrack(test, csp)  # Recurse with the new assignment
            if result[0]:
                return result
    return False, assignments  # No solution found, return failure


# Function to convert the assignment dictionary back into a Sudoku board string
def board_to_string(assignment, csp):
    rows = "ABCDEFGHI"
    cols = "123456789"
    output = ""

    # Build the output string from the board
    for r in rows:
        for c in cols:
            if r + c not in assignment:  # If the cell isn't assigned yet
                output += str(csp.board[r + c])
            else:
                output += str(assignment[r + c])  # Add the assigned value

    return output


# Inputs: List of compressed Sudoku boards as strings
inputs = ["000000000302540000050301070000000004409006005023054790000000050700810000080060009",
          "000260701680070090190004500820100040004602900050003028009300074040050036703018000",
          "000100702030950000001002003590000301020000070703000098800200100000085060605009000"]

# Loop through the input boards, solve them, and print the results
for item in inputs:
    csp = CSP(item)  # Create a CSP instance from the input board
    AC3(csp)  # Enforce arc consistency
    assignment = BTS(csp)  # Solve the board using backtracking
    solved_board = board_to_string(assignment, csp)  # Convert the assignment to a string
    print_board(csp)  # Print the original board
    print("Solved Board:")
    print(solved_board)  # Print the solved board
    print()
