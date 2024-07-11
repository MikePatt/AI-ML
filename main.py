from collections import Counter
from operator import itemgetter
from copy import copy


class CSP:
    def __init__(self, initBoard):
        self.board = {}
        self.domain = {}
        self.constraints = []
        self.rows = "ABCDEFGHI"
        self.columns = "123456789"

        for i, c in enumerate(self.rows):
            for j, d in enumerate(self.columns):

                key = c + d
                offset = i * 8
                value = int(initBoard[i + j + offset])

                # Add to Board
                self.board[key] = value

                # Add to Domain
                if value == 0:
                    self.domain[key] = [i + 1 for i in range(9)]
                else:
                    self.domain[key] = [value]

        # append to Constraint
        for i in self.rows:
            self.constraints.append([i + j for j in self.columns])

        for i in self.columns:
            self.constraints.append([j + i for j in self.rows])

        a = [[[i + j for j in self.columns[0 + k:3 + k]] for i in self.rows] for k in range(0, 9, 3)]
        for i in range(3):
            for k in range(0, 9, 3):
                self.constraints.append([item for sublist in a[i][0 + k:3 + k] for item in sublist])


def print_board(csp):
    rows = "ABCDEFGHI"
    cols = "123456789"
    for i, r in enumerate(rows):
        if i in [3, 6]:
            print('------+-------+------'),
        for j, c in enumerate(cols):
            if j in [3, 6]:
                print('|', end=" "),
            print(csp.board[r + c], end=" ")
        print()


def print_solved_domain(csp):
    # Returns the printed solution if the domain of each variable has only one element
    # Otherwise prints the AC3 domain.
    if max([len(val) for val in csp.domain.values()]) == 1:
        solution_string = "".join([str(item) for sublist in list(csp.domain.values()) for item in sublist])
        print("Sudoku is Solved")
        print()
    else:
        print("Sudoku is Not Solved")
        print()
        print("Domain: ")
        print(csp.domain)


def CreateArcs(constraints):
    arcs_queue = []
    for constraint in constraints:
        for x in constraint:
            for y in constraint:
                if x != y:
                    arcs_queue.append((x, y))
    return arcs_queue


def Revise(csp, Xi, Xj):
    revised = False
    for Di in csp.domain[Xi]:
        satisfied = False
        for Dj in csp.domain[Xj]:
            if Di != Dj:
                satisfied = True
        if not satisfied:
            csp.domain[Xi].remove(Di)
            revised = True
    return revised


def AC3(csp):
    arcs_queue = CreateArcs(csp.constraints)
    fringe = arcs_queue.copy()
    while fringe:
        (Xi, Xj) = fringe.pop(0)
        if Revise(csp, Xi, Xj):
            if len(csp.domain[Xi]) == 0:
                return False
            re_arcs = [arc for arc in arcs_queue if arc[1] == Xi and arc[0] != Xj]
            fringe.extend(re_arcs)
    return True


def SelectUnassignVariable(assignments, csp):
    return min([item for item in csp.domain if item not in assignments.keys() and csp.board[item] == 0], key=lambda x: len(csp.domain[x]))

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

def CheckConsistent(assignment, csp):
    varis = [key for key in assignment.keys()]
    for var in varis:
        for constraint in csp.constraints:
            if var in constraint:
                peers = [item for item in constraint if item != var and item in assignment.keys()]
                for peer in peers:
                    if assignment[peer] == assignment[var]:
                        return False
    return True

def BTS(csp):
    answer, assignment = BackTrack({}, csp)
    return assignment

def BackTrack(assignments, csp):
    if CheckAssignment(assignments, csp):
        return True, assignments
    var = SelectUnassignVariable(assignments, csp)
    for value in csp.domain[var]:
        test = assignments.copy()
        test[var] = value
        consistent = CheckConsistent(test, csp)
        if consistent:
            result = BackTrack(test, csp)
            if result[0]:
                return result
    return False, assignments


def board_to_string(assignment, csp):
    rows = "ABCDEFGHI"
    cols = "123456789"
    output = ""

    for r in rows:
        for c in cols:
            if r + c not in assignment:
                output += str(csp.board[r + c])
            else:
                output += str(assignment[r + c])

    return output


inputs = ["000000000302540000050301070000000004409006005023054790000000050700810000080060009", "000260701680070090190004500820100040004602900050003028009300074040050036703018000", "000100702030950000001002003590000301020000070703000098800200100000085060605009000"]
for item in inputs:
    csp = CSP(item)
    AC3(csp)
    assignment = BTS(csp)
    solved_board = board_to_string(assignment, csp)
    print_board(csp)
    print("Solved Board:")
    print(solved_board)
    print()