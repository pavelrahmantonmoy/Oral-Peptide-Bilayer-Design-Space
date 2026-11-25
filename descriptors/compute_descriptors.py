"""
Recomputes every value in Table 3 of the manuscript.

The Tirzepatide structure is the complete 39-residue sequence including both
alpha-aminoisobutyric acid residues and the gamma-glutamate-bis(AEEA)-linked C20
fatty diacid.  Before any descriptor is computed the parsed structure is checked
against the published molecular formula C225H348N48O68; the script aborts if it
does not match, so a truncated or mis-transcribed SMILES cannot silently produce
descriptors.

Run:  python3 -I compute_descriptors.py
"""
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski

EXPECTED_FORMULA = "C225H348N48O68"

def load(path):
    smi = open(path).read().strip()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        sys.exit(f"ERROR: {path} could not be parsed")
    return m

def describe(name, m):
    print(f"\n{name}")
    print(f"  Molecular formula      {rdMolDescriptors.CalcMolFormula(m)}")
    print(f"  Molecular weight       {Descriptors.MolWt(m):.2f} g/mol")
    print(f"  logP (Wildman-Crippen) {Crippen.MolLogP(m):+.2f}")
    print(f"  TPSA                   {rdMolDescriptors.CalcTPSA(m):.2f} A^2")
    print(f"  H-bond donors          {Lipinski.NumHDonors(m)}")
    print(f"  H-bond acceptors       {Lipinski.NumHAcceptors(m)}")
    print(f"  Rotatable bonds        {Lipinski.NumRotatableBonds(m)}")
    print(f"  Heavy atoms            {m.GetNumHeavyAtoms()}")

tz = load("tirzepatide.smi")
formula = rdMolDescriptors.CalcMolFormula(tz)
if formula != EXPECTED_FORMULA:
    sys.exit(f"ERROR: formula {formula} does not match published {EXPECTED_FORMULA}; "
             "the structure is incomplete or mis-transcribed. Aborting.")
print(f"Structure check passed: {formula} == {EXPECTED_FORMULA}")

describe("Tirzepatide", tz)
describe("SNAC (free acid, salcaprozic acid)", load("snac_free_acid.smi"))
describe("SNAC (sodium salt, salcaprozate sodium)", load("snac_sodium.smi"))

print("""
Note on interpretation
----------------------
The Wildman-Crippen logP is an additive fragment estimate parameterised on small
organic molecules.  For a 4.81 kDa peptide it is reported as a formal descriptor
indicating the direction and magnitude of hydrophilicity, not as a measured
partition coefficient.  These descriptors are read within the beyond-rule-of-five
(bRo5) framework: they quantify the permeability barrier to be overcome by
formulation, they are not a Lipinski pass/fail filter, and no fingerprint
similarity between the peptide and the enhancer is computed (see Section 2.5).
""")
