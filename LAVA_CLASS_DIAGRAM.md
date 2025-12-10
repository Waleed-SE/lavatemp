# LAVA Framework - Complete Class Diagram & Structure

## Table of Contents

1. [Class Hierarchy Overview](#class-hierarchy-overview)
2. [Core Classes Detailed](#core-classes-detailed)
3. [Interface & Implementation Classes](#interface--implementation-classes)
4. [Supporting Classes](#supporting-classes)
5. [Method Signature Reference](#method-signature-reference)
6. [Module Dependencies](#module-dependencies)

---

## Class Hierarchy Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          LAVA Framework                             │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      LAVA (Root)                             │  │
│  │  Manages projects and coordinates vulnerability analysis    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│                              │                                       │
│                              ├── Projects: List[Project]            │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      Project                                 │  │
│  │  Core analysis unit for a smart contract                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │          │            │              │                    │
│         │          │            │              └─────── Solver      │
│         │          │            │                                    │
│         │          ├─── AST     │                                    │
│         │          │            │                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Facts Interface         │    Rules Interface                │  │
│  │                          │                                    │  │
│  │  Implementations:        │    Implementations:               │  │
│  │  • ReentrancyFacts       │    • ReentrancyRules              │  │
│  │  • OverflowFacts         │    • OverflowRules                │  │
│  │  • DelegateCallFacts     │    • DelegateCallRules            │  │
│  │  • SelfDestructFacts     │    • SelfDestructRules            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Classes Detailed

### 1. LAVA (Root Class)

**Package:** `lava.py`  
**Type:** Root orchestrator class  
**Database Enabled:** ✅ Yes

#### Attributes

```python
@classmethod
LAVA:
    # Public Attributes
    - projects: List[Project]           # List of all managed projects
    - use_database: bool                # Enable/disable database persistence
```

#### Methods

| Method                | Signature                         | Returns         | Description                                      |
| --------------------- | --------------------------------- | --------------- | ------------------------------------------------ |
| `__init__()`          | `(use_database: bool = True)`     | `None`          | Initialize LAVA framework with optional database |
| `create_project()`    | `(name: str) → Project`           | `Project`       | Create new project and add to projects list      |
| `add_project()`       | `(project: Project) → None`       | `None`          | Add existing project to framework                |
| `get_project()`       | `(name: str) → Optional[Project]` | `Project\|None` | Retrieve project by name                         |
| `remove_project()`    | `(name: str) → bool`              | `bool`          | Remove project from memory and database          |
| `load_all_projects()` | `() → None`                       | `None`          | Load all projects from database                  |
| `save_all_projects()` | `() → None`                       | `None`          | Save all projects to database                    |
| `get_ast()`           | `(project: Project) → AST`        | `AST`           | Get AST from specific project                    |
| `main()`              | `(argv: List[str]) → None`        | `None`          | CLI entry point                                  |
| `__repr__()`          | `() → str`                        | `str`           | String representation                            |

#### Example Usage

```python
# Create LAVA instance
lava = LAVA(use_database=True)

# Create and add project
project = Project(name="MyContract")
lava.add_project(project)

# Retrieve project
project = lava.get_project("MyContract")

# Load all projects from DB
lava.load_all_projects()

# Save all projects to DB
lava.save_all_projects()
```

---

### 2. Project

**Package:** `lava.py`  
**Type:** Dataclass  
**Database Enabled:** ✅ Yes  
**Decorators:** `@dataclass`

#### Attributes

```python
@dataclass
class Project:
    # Core Attributes
    - name: str                                           # Project name
    - db_id: Optional[str] = None                        # Database ID for persistence

    # Private Components (Lazy Initialized)
    - _reentrancy_facts: Optional[ReentrancyFacts]
    - _integer_overflow_and_underflow_facts: Optional[IntegerOverflowAndUnderflowFacts]
    - _delegatecall_facts: Optional[DelegateCallFacts]
    - _selfdestruct_facts: Optional[SelfDestructFacts]

    - _reentrancy_rules: Optional[ReentrancyRules]
    - _integer_overflow_and_underflow_rules: Optional[IntegerOverflowAndUnderflowRules]
    - _delegatecall_rules: Optional[DelegateCallRules]
    - _selfdestruct_rules: Optional[SelfDestructRules]

    - _solver: Optional[Solver]                          # ASP solver
    - _ast: Optional[AST]                                # Abstract syntax tree
```

#### Methods

| Method                      | Signature                                    | Returns         | Description                                         |
| --------------------------- | -------------------------------------------- | --------------- | --------------------------------------------------- |
| `__post_init__()`           | `() → None`                                  | `None`          | Initialize all components after dataclass creation  |
| `get_facts()`               | `(kind: str) → Facts`                        | `Facts`         | Get facts generator for vulnerability type          |
| `get_rules()`               | `(kind: str) → Rules`                        | `Rules`         | Get rules generator for vulnerability type          |
| `get_solver()`              | `() → Solver`                                | `Solver`        | Get solver instance                                 |
| `get_ast()`                 | `() → AST`                                   | `AST`           | Get AST instance                                    |
| `set_llm_provider()`        | `(provider_name: str) → None`                | `None`          | Set LLM provider for rule generation                |
| `analyze_vulnerabilities()` | `(vulnerability_type: str) → Dict[str, Any]` | `Dict`          | Full analysis workflow (Code→AST→Facts→Rules→Solve) |
| `solve_with_existing()`     | `(vulnerability_type: str) → Dict[str, Any]` | `Dict`          | Solve using existing facts and rules                |
| `save_to_db()`              | `() → bool`                                  | `bool`          | Save project to database                            |
| `_update_component_ids()`   | `() → None`                                  | `None`          | Update project_id in all components                 |
| `load_from_db()`            | `(project_id: str) → Optional[Project]`      | `Project\|None` | Static method to load project from database         |
| `__repr__()`                | `() → str`                                   | `str`           | String representation                               |

#### Vulnerability Types Supported

- `"reentrancy"`
- `"integer_overflow"` or `"overflow"`
- `"delegatecall"`
- `"selfdestruct"`

#### Example Usage

```python
# Create project
project = Project(name="TimeLock")

# Get components
ast = project.get_ast()
reentrancy_facts = project.get_facts("reentrancy")
reentrancy_rules = project.get_rules("reentrancy")
solver = project.get_solver()

# Set LLM provider
project.set_llm_provider("openai")

# Perform full analysis
results = project.analyze_vulnerabilities("reentrancy")

# Solve with existing facts/rules
results = project.solve_with_existing("integer_overflow")

# Save to database
project.save_to_db()

# Load from database
loaded_project = Project.load_from_db(project.db_id)
```

---

### 3. AST (Abstract Syntax Tree)

**Package:** `lava.py`  
**Type:** Regular class  
**Database Enabled:** ✅ Yes

#### Attributes

```python
class AST:
    # Core Attributes
    - project_id: Optional[str]                    # Reference to parent project
    - _ast_json: Optional[Dict]                    # AST JSON structure
    - solidity_code: Optional[str]                 # Solidity source code
    - db_id: Optional[str]                         # Database ID for persistence
```

#### Methods

| Method                           | Signature                            | Returns       | Description                             |
| -------------------------------- | ------------------------------------ | ------------- | --------------------------------------- |
| `__init__()`                     | `(project_id: Optional[str] = None)` | `None`        | Initialize empty AST                    |
| `get_ast()`                      | `() → Dict`                          | `Dict`        | Get AST JSON structure                  |
| `set_ast()`                      | `(ast_json: Dict) → None`            | `None`        | Set AST and save to database            |
| `get_solidity_code()`            | `() → Optional[str]`                 | `str\|None`   | Get Solidity source code                |
| `set_solidity_code()`            | `(solidity_code: str) → None`        | `None`        | Set Solidity code and save to database  |
| `compile_from_solidity_code()`   | `(code: str) → Tuple[bool, str]`     | `(bool, str)` | Compile Solidity to JSON AST using solc |
| `generate_facts()`               | `() → None`                          | `None`        | Generate facts from AST                 |
| `generate_reentrancy_code()`     | `() → None`                          | `None`        | Generate reentrancy-specific code       |
| `get_overflow_underflow_facts()` | `() → List[str]`                     | `List[str]`   | Extract overflow/underflow facts        |
| `get_reentrancy_facts()`         | `() → List[str]`                     | `List[str]`   | Extract reentrancy facts                |
| `save_to_db()`                   | `() → str`                           | `str`         | Save to database, return ID             |
| `load_from_db()`                 | `() → bool`                          | `bool`        | Load from database using project_id     |
| `update_in_db()`                 | `() → bool`                          | `bool`        | Update existing record in database      |
| `delete_from_db()`               | `() → bool`                          | `bool`        | Delete from database                    |

#### Example Usage

```python
# Create AST
ast = AST(project_id="proj123")

# Set Solidity code
ast.set_solidity_code("contract MyContract { ... }")

# Compile from Solidity
success, result = ast.compile_from_solidity_code("contract MyContract { ... }")

# Get AST and code
ast_dict = ast.get_ast()
code = ast.get_solidity_code()

# Save to database
ast.save_to_db()

# Load from database
ast.load_from_db()

# Get vulnerability-specific facts
overflow_facts = ast.get_overflow_underflow_facts()
reentrancy_facts = ast.get_reentrancy_facts()
```

---

### 4. Solver

**Package:** `lava.py`  
**Type:** Regular class  
**Integration:** Clingo ASP solver

#### Attributes

```python
class Solver:
    # Core Attributes
    - project_id: Optional[str]                    # Reference to parent project
    - _facts: Optional[Facts]                      # Last used facts
    - _rules: Optional[Rules]                      # Last used rules
    - _clingo_result: Optional[Dict]               # Clingo solver output
```

#### Methods

| Method                  | Signature                                   | Returns       | Description                                |
| ----------------------- | ------------------------------------------- | ------------- | ------------------------------------------ |
| `__init__()`            | `(project_id: Optional[str] = None)`        | `None`        | Initialize empty solver                    |
| `solve()`               | `(facts: Facts, rules: Rules) → List[bool]` | `List[bool]`  | Execute Clingo solver with facts and rules |
| `get_model()`           | `() → List[bool]`                           | `List[bool]`  | Get current model/solution                 |
| `get_facts()`           | `() → Optional[Facts]`                      | `Facts\|None` | Get facts from last solve                  |
| `get_rules()`           | `() → Optional[Rules]`                      | `Rules\|None` | Get rules from last solve                  |
| `get_clingo_result()`   | `() → Optional[Dict]`                       | `Dict\|None`  | Get raw Clingo result                      |
| `get_answer_sets()`     | `() → List[Dict[str, Any]]`                 | `List[Dict]`  | Get answer sets from last solve            |
| `get_vulnerabilities()` | `() → List[Dict[str, Any]]`                 | `List[Dict]`  | Extract vulnerabilities from answer sets   |
| `is_satisfiable()`      | `() → bool`                                 | `bool`        | Check if last solve was satisfiable        |
| `get_status()`          | `() → Dict[str, Any]`                       | `Dict`        | Get solver status summary                  |

#### Example Usage

```python
# Create solver
solver = Solver(project_id="proj123")

# Solve with facts and rules
facts = project.get_facts("reentrancy")
rules = project.get_rules("reentrancy")
solver.solve(facts, rules)

# Get results
is_sat = solver.is_satisfiable()
answer_sets = solver.get_answer_sets()
vulnerabilities = solver.get_vulnerabilities()
status = solver.get_status()

# Check last used facts/rules
last_facts = solver.get_facts()
last_rules = solver.get_rules()
```

---

## Interface & Implementation Classes

### Facts Interface (Abstract Base Class)

**Package:** `lava.py`  
**Type:** Abstract Base Class (ABC)  
**Superclass:** None

#### Abstract Methods

```python
@abstractmethod
class Facts:

    def get_facts() → List[str]:
        """Return the list of generated facts."""
        raise NotImplementedError

    def generate_facts(ast: 'AST') → None:
        """Generate facts from the given AST."""
        raise NotImplementedError
```

---

### Concrete Facts Implementations

#### 1. ReentrancyFacts

**Package:** `lava.py`  
**Type:** Dataclass  
**Extends:** `Facts`  
**Database Enabled:** ✅ Yes

```python
@dataclass
class ReentrancyFacts(Facts):
    # Attributes
    - facts: List[str] = field(default_factory=list)
    - project_id: Optional[str] = None
    - db_id: Optional[str] = None
    - llm_provider: Optional[str] = None

    # Methods
    + get_facts() → List[str]
    + generate_facts(ast: AST) → None
    + __repr__() → str
```

#### 2. IntegerOverflowAndUnderflowFacts

**Package:** `lava.py`  
**Type:** Dataclass  
**Extends:** `Facts`  
**Database Enabled:** ✅ Yes

```python
@dataclass
class IntegerOverflowAndUnderflowFacts(Facts):
    # Attributes
    - facts: List[str] = field(default_factory=list)
    - project_id: Optional[str] = None
    - db_id: Optional[str] = None

    # Methods
    + get_facts() → List[str]
    + generate_facts(ast: AST) → None
    + __repr__() → str
```

#### 3. DelegateCallFacts

**Package:** `lava.py`  
**Type:** Dataclass  
**Extends:** `Facts`  
**Database Enabled:** ✅ Yes

```python
@dataclass
class DelegateCallFacts(Facts):
    # Attributes
    - facts: List[str] = field(default_factory=list)
    - project_id: Optional[str] = None
    - db_id: Optional[str] = None

    # Methods
    + get_facts() → List[str]
    + generate_facts(ast: AST) → None
    + __repr__() → str
```

#### 4. SelfDestructFacts

**Package:** `lava.py`  
**Type:** Dataclass  
**Extends:** `Facts`  
**Database Enabled:** ✅ Yes

```python
@dataclass
class SelfDestructFacts(Facts):
    # Attributes
    - facts: List[str] = field(default_factory=list)
    - project_id: Optional[str] = None
    - db_id: Optional[str] = None

    # Methods
    + get_facts() → List[str]
    + generate_facts(ast: AST) → None
    + __repr__() → str
```

---

### Rules Interface (Abstract Base Class)

**Package:** `lava.py`  
**Type:** Abstract Base Class (ABC)

#### Abstract Methods

```python
@abstractmethod
class Rules:

    def get_rules() → List[str]:
        """Return the list of generated rules."""
        raise NotImplementedError

    def generate_rules() → None:
        """Generate vulnerability detection rules."""
        raise NotImplementedError
```

---

### Concrete Rules Implementations

#### 1. ReentrancyRules

**Package:** `lava.py`  
**Type:** Dataclass  
**Extends:** `Rules`  
**Generator:** LLM-based  
**Database Enabled:** ✅ Yes

```python
@dataclass
class ReentrancyRules(Rules):
    # Attributes
    - rules: List[str] = field(default_factory=list)
    - project_id: Optional[str] = None
    - db_id: Optional[str] = None
    - llm_provider: Optional[str] = None

    # Methods
    + get_rules() → List[str]
    + generate_rules(ast: Optional[AST] = None, facts: Optional[List[str]] = None) → None
    + _get_template_rules() → List[str]
    + __repr__() → str
```

#### 2. IntegerOverflowAndUnderflowRules

**Package:** `lava.py`  
**Type:** Dataclass  
**Extends:** `Rules`  
**Generator:** LLM-based  
**Database Enabled:** ✅ Yes

```python
@dataclass
class IntegerOverflowAndUnderflowRules(Rules):
    # Attributes
    - rules: List[str] = field(default_factory=list)
    - project_id: Optional[str] = None
    - db_id: Optional[str] = None
    - llm_provider: Optional[str] = None

    # Methods
    + get_rules() → List[str]
    + generate_rules(ast: Optional[AST] = None, facts: Optional[List[str]] = None) → None
    + _get_template_rules() → List[str]
    + __repr__() → str
```

#### 3. DelegateCallRules

**Package:** `lava.py`  
**Type:** Dataclass  
**Extends:** `Rules`  
**Generator:** LLM-based  
**Database Enabled:** ✅ Yes

```python
@dataclass
class DelegateCallRules(Rules):
    # Attributes
    - rules: List[str] = field(default_factory=list)
    - project_id: Optional[str] = None
    - db_id: Optional[str] = None
    - llm_provider: Optional[str] = None

    # Methods
    + get_rules() → List[str]
    + generate_rules(ast: Optional[AST] = None, facts: Optional[List[str]] = None) → None
    + __repr__() → str
```

#### 4. SelfDestructRules

**Package:** `lava.py`  
**Type:** Dataclass  
**Extends:** `Rules`  
**Generator:** LLM-based  
**Database Enabled:** ✅ Yes

```python
@dataclass
class SelfDestructRules(Rules):
    # Attributes
    - rules: List[str] = field(default_factory=list)
    - project_id: Optional[str] = None
    - db_id: Optional[str] = None
    - llm_provider: Optional[str] = None

    # Methods
    + get_rules() → List[str]
    + generate_rules(ast: Optional[AST] = None, facts: Optional[List[str]] = None) → None
    + __repr__() → str
```

---

## Supporting Classes

### Database Models

**Package:** `database.py`  
**Type:** Database ORM Models

#### Core Models

1. **ProjectModel** - Manages project records
2. **FactsModel** - Manages facts records
3. **RulesModel** - Manages rules records
4. **ASTModel** - Manages AST records
5. **SolverModel** - Manages solver state records

#### Key Functions

```python
# Database convenience functions
save_facts_to_db(project_id, vulnerability_type, facts, db_id) → str
load_facts_from_db(project_id, vulnerability_type) → (List[str], Optional[str])

save_rules_to_db(project_id, vulnerability_type, rules, db_id) → str
load_rules_from_db(project_id, vulnerability_type) → (List[str], Optional[str])

save_ast_to_db(project_id, ast_json, solidity_code, db_id) → str
load_ast_from_db(project_id) → (Dict, Optional[str], Optional[str])
delete_ast_from_db(project_id) → bool

save_project_to_db(name, db_id) → str
load_solver_state_from_db(project_id) → Optional[Dict]
```

---

### AST Parser Classes

**Package:** `ast_parser.py`

```python
class SolidityASTParser:
    + parse_ast(ast_dict, project_name) → str
    + extract_facts() → List[str]

class ReentrancyASTParser(SolidityASTParser):
    + parse_ast(ast_dict, project_name) → str
    + analyze_reentrancy() → List[str]

def generate_asp_facts_from_dict(ast_dict) → str:
    """Convert AST dict to Answer Set Programming facts"""
```

---

### Solver Classes

**Package:** `clingo_solver.py`

```python
class ClingoSolver:
    + solve(facts: List[str], rules: List[str]) → Dict
    + get_answer_sets() → List[str]
    + is_satisfiable() → bool
    + extract_vulnerabilities() → List[Dict]
```

---

### Solidity Compiler

**Package:** `solc_compiler.py`

```python
class SolidityCompiler:
    + compile(code: str) → Dict  # Returns JSON AST
    + get_ast() → Dict
    + filter_ast(ast: Dict) → Dict
```

---

### LLM Provider

**Package:** `llm_provider.py`

```python
class LLMFactory:
    + create_provider(name: str) → LLMProvider
    + get_available_providers() → List[str]

def generate_rules(
    vulnerability_type: str,
    facts: List[str],
    solidity_code: Optional[str],
    llm_provider: Optional[str],
    use_llm: bool = True
) → str:
    """Generate ASP rules using LLM"""

def generate_facts(
    ast_json: Dict,
    vulnerability_type: str
) → List[str]:
    """Generate facts from AST"""
```

---

## Method Signature Reference

### Facts Generation Workflow

```python
# Step 1: Get facts generator
facts = project.get_facts("reentrancy")

# Step 2: Generate facts from AST
facts.generate_facts(ast)

# Step 3: Get generated facts list
facts_list = facts.get_facts()  # Returns List[str]

# Step 4: Save to database
facts.db_id = save_facts_to_db(project.db_id, "reentrancy", facts_list, facts.db_id)
```

### Rules Generation Workflow

```python
# Step 1: Get rules generator
rules = project.get_rules("reentrancy")

# Step 2: Generate rules using LLM
rules.generate_rules(ast=project.get_ast(), facts=facts_list)

# Step 3: Get generated rules list
rules_list = rules.get_rules()  # Returns List[str]

# Step 4: Save to database (automatic in generate_rules)
# or manual: save_rules_to_db(project.db_id, "reentrancy", rules_list, rules.db_id)
```

### Solving Workflow

```python
# Step 1: Get solver
solver = project.get_solver()

# Step 2: Execute solver with facts and rules
solver.solve(facts, rules)

# Step 3: Get results
is_satisfiable = solver.is_satisfiable()
answer_sets = solver.get_answer_sets()
vulnerabilities = solver.get_vulnerabilities()

# Step 4: Check status
status = solver.get_status()
print(f"Status: {status}")
```

### Complete Analysis Workflow

```python
# Method 1: Full Workflow (Code → AST → Facts → Rules → Solve)
results = project.analyze_vulnerabilities("reentrancy")
# Returns: {
#     "vulnerability_type": str,
#     "satisfiable": bool,
#     "answer_sets": List[Dict],
#     "vulnerabilities": List[Dict],
#     "facts_count": int,
#     "rules_count": int,
#     "workflow": str
# }

# Method 2: Solve with Existing (Facts + Rules → Solve)
results = project.solve_with_existing("reentrancy")
# Returns same structure as above
```

---

## Module Dependencies

```
lava.py (Core Framework)
  ├── database.py (Persistence)
  ├── ast_parser.py (AST Parsing)
  ├── clingo_solver.py (Clingo Integration)
  ├── solc_compiler.py (Solidity Compilation)
  ├── llm_provider.py (LLM Integration)
  │   ├── OpenAI (optional)
  │   └── Google Generative AI (optional)
  └── ast_filter.py (AST Filtering)

lava_gui.py (Streamlit Interface)
  └── lava.py (Core)
     └── (all dependencies above)
```

---

## Database Schema Summary

### Collections

#### projects

```
{
  _id: ObjectId,
  name: str,
  created_at: datetime,
  updated_at: datetime
}
```

#### facts

```
{
  _id: ObjectId,
  project_id: ObjectId,
  vulnerability_type: str,
  facts: [str],
  created_at: datetime,
  updated_at: datetime
}
```

#### rules

```
{
  _id: ObjectId,
  project_id: ObjectId,
  vulnerability_type: str,
  rules: [str],
  created_at: datetime,
  updated_at: datetime
}
```

#### asts

```
{
  _id: ObjectId,
  project_id: ObjectId,
  ast_json: dict,
  solidity_code: str,
  created_at: datetime,
  updated_at: datetime
}
```

#### solver_results

```
{
  _id: ObjectId,
  project_id: ObjectId,
  vulnerability_type: str,
  result: dict,
  created_at: datetime
}
```

---

## Code Statistics

| Component                | Type | Count                           |
| ------------------------ | ---- | ------------------------------- |
| Core Classes             | 4    | LAVA, Project, AST, Solver      |
| Abstract Classes         | 2    | Facts, Rules                    |
| Concrete Implementations | 8    | 4 Facts + 4 Rules               |
| Supporting Classes       | 10+  | Database, Parser, Compiler, LLM |
| **Total**                |      | **24+**                         |

---

## Usage Examples

### Example 1: Basic Project Creation and Analysis

```python
from lava import LAVA, Project

# Initialize framework
lava = LAVA(use_database=True)

# Create project
project = Project(name="MySmartContract")
lava.add_project(project)

# Compile Solidity to AST
ast = project.get_ast()
success, result = ast.compile_from_solidity_code(solidity_code)

# Perform full analysis
results = project.analyze_vulnerabilities("reentrancy")

# Check results
if results['satisfiable']:
    print(f"Found {len(results['vulnerabilities'])} vulnerabilities")
```

### Example 2: Manual Step-by-Step Analysis

```python
# Get components
facts_gen = project.get_facts("integer_overflow")
rules_gen = project.get_rules("integer_overflow")
solver = project.get_solver()
ast = project.get_ast()

# Step 1: Generate facts
facts_gen.generate_facts(ast)
facts_list = facts_gen.get_facts()

# Step 2: Generate rules
rules_gen.generate_rules(ast=ast, facts=facts_list)
rules_list = rules_gen.get_rules()

# Step 3: Solve
solver.solve(facts_gen, rules_gen)

# Step 4: Get results
answer_sets = solver.get_answer_sets()
vulnerabilities = solver.get_vulnerabilities()
```

### Example 3: Working with Database

```python
# Save project
project.save_to_db()

# Load all projects
lava.load_all_projects()

# Load specific project
loaded = Project.load_from_db(project.db_id)

# Remove project (cascade delete)
lava.remove_project("MySmartContract")
```

---

## Key Features

✅ **Type-Safe** - Uses Python dataclasses and type hints  
✅ **Database Persistence** - MongoDB integration with CRUD operations  
✅ **LLM Integration** - OpenAI and Google Generative AI support  
✅ **ASP Solving** - Clingo integration for vulnerability detection  
✅ **AST Filtering** - Optimized AST processing (62.9% size reduction)  
✅ **Multi-Vulnerability** - 4 vulnerability types supported  
✅ **GUI Interface** - Streamlit-based user interface  
✅ **Modular Design** - Interfaces for easy extension

---

**Generated:** December 10, 2025  
**LAVA Version:** 1.0  
**Status:** Production Ready ✅
