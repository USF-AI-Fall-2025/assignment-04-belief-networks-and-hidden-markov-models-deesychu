# Adrienne Dominique Sy Chu

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = DiscreteBayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
])

# Defining the parameters using CPT

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))


def main():
    print("\n--- Car Network Queries ---")

    # Q1
    q1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print("\nQ1. P(Battery | Moves=no):\n", q1)

    # Q2
    q2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print("\nQ2. P(Starts | Radio=Doesn't turn on):\n", q2)

    # Q3
    q3a = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    q3b = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print("\nQ3a. P(Radio | Battery=Works):\n", q3a)
    print("\nQ3b. P(Radio | Battery=Works, Gas=Full):\n", q3b)

    # Q4
    q4a = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    q4b = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print("\nQ4a. P(Ignition | Moves=no):\n", q4a)
    print("\nQ4b. P(Ignition | Moves=no, Gas=Empty):\n", q4b)

    # Q5
    q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print("\nQ5. P(Starts | Radio=turns on, Gas=Full):\n", q5)

    # Q6 (with KeyPresent)
    car_model.add_node("KeyPresent")
    car_model.add_edge("KeyPresent", "Starts")

    cpd_key = TabularCPD(
        variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
        state_names={"KeyPresent": ["yes", "no"]}
    )

    cpd_starts_new = TabularCPD(
        variable="Starts", variable_card=2,
        values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
        evidence=["Gas", "Ignition", "KeyPresent"],
        evidence_card=[2, 2, 2],
        state_names={"Starts": ['yes', 'no'],
                    "Gas": ['Full', 'Empty'],
                    "Ignition": ["Works", "Doesn't work"],
                    "KeyPresent": ["yes", "no"]}
    )

    car_model.add_cpds(cpd_key, cpd_starts_new)
    car_infer2 = VariableElimination(car_model)
    q6 = car_infer2.query(variables=["KeyPresent"], evidence={"Moves": "no"})
    print("\nQ6. P(KeyPresent | Moves=no):\n", q6)

if __name__ == "__main__":
    main()