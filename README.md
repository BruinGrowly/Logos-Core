# Project Logos: The Semantic Overlay

**Version:** 0.1 (Genesis)
**Core Axiom:** "Code with Meaning, not just Instruction."

## 1. Project Overview

**Logos** is a prototype Semantic Operating System that sits atop Microsoft Windows. It creates a "Meaning Layer" between the user and the file system.

* **The Language (.logos):** A declarative syntax to define the *Context*, *Intent*, and *Gravity* of a folder or project.
* **The OS (Overlay):** A Python-based runtime that watches the user's activity, reads `.logos` files, and enforces semantic rules (e.g., "Deep Work Mode", "Auto-Organization").

## 2. Architecture Diagram

* **Layer 1: The Substrate (Windows):** Holds raw files (`.txt`, `.pdf`).
* **Layer 2: The Membrane (Python Runtime):**
* **Sensors:** Monitors file system changes (`watchdog`).
* **Interpreter:** Parses `.logos` files.
* **Memory:** Vector Database (ChromaDB) for semantic indexing.


* **Layer 3: The Spirit (LLM):** Local inference (Ollama) or API to provide "Reasoning."

---

## 3. The "Logos" Language Syntax (v0.1)

The AI Developer needs to build a parser for this specific syntax. It is **YAML-based** but **Entity-Oriented**.

**File Extension:** `project.logos`

```yaml
# EXAMPLE SYNTAX v0.1

Manifest: Project_Alpha
Type: Entity
State: Active

Context:
  Goal: "Complete the Q1 Financial Report"
  Gravity: High                 # Determines priority/sorting
  Owner: "Formalizer"
  Vectors: [Finance, Strategy, Confidential]

Rules:
  # The "Conscience" of the folder
  - If: "New File" aligns_with "Receipts"
    Then: "Move to /Expenses folder"

  - If: "User Focus" drops_below 50%
    Then: "Alert 'Return to the Anchor'"

Constraints:
  - Reject: "Social Media Links"
  - Enforce: "Encryption Level 5"

```

---

## 4. Implementation Plan (Phased)

### Phase 1: The "Spinal Cord" (The Interpreter)

**Objective:** Build a Python script that can read a `.logos` file and understand it.

* **Task 1.1:** Create `interpreter.py`.
* **Task 1.2:** Implement a parser (using `PyYAML`) to ingest `project.logos`.
* **Task 1.3:** Create a simple "Action Engine" that prints the intent to the console.
* *Input:* `project.logos`
* *Output:* "System acknowledges: Gravity is HIGH. Goal is FINANCIAL REPORT."



### Phase 2: The "Eyes" (The Watcher)

**Objective:** Make the OS react when files move.

* **Task 2.1:** Install `watchdog` library.
* **Task 2.2:** Create `watcher.py` to monitor a specific "Test Folder."
* **Task 2.3:** Connect Watcher to Interpreter.
* *Logic:* When a file is dropped in "Test Folder," read the local `.logos` file to decide what to do with it.



### Phase 3: The "Brain" (Semantic Routing)

**Objective:** Connect to an LLM to perform "Fuzzy Logic."

* **Task 3.1:** Integrate `ollama` or `openai` library.
* **Task 3.2:** Implement the `aligns_with` function.
* *Logic:* Instead of checking if file extension is `.pdf`, ask the LLM: *"Does this file content align with the concept of 'Receipts'?"*



---

## 5. Directory Structure for the Repo

```text
/logos-core
│
├── /cortex          (Logic Core)
│   ├── __init__.py
│   ├── interpreter.py   # The Logos Parser
│   └── llm_bridge.py    # Connection to AI Model
│
├── /sensory         (Input/Output)
│   ├── watcher.py       # File System Monitor
│   └── tray_icon.py     # System Tray UI (Cyan Pulse)
│
├── /memory          (State)
│   └── vector_store.py  # ChromaDB Integration
│
├── main.py          # Entry Point (Run this to start the OS)
├── requirements.txt # Dependencies (watchdog, pyyaml, langchain, etc.)
└── README.md        # This Plan

```

---

## 6. Instruction to the AI Developer

*(Copy and paste this into the first issue or prompt for the AI)*

> **Prompt:**
> "Act as a Senior Systems Architect. We are building 'Logos,' a semantic overlay for Windows.
> **Goal:** Create a Python application that monitors a folder. When it detects a `project.logos` file (YAML format), it parses the rules defined therein.
> **Step 1:** Write the `interpreter.py` that parses the YAML syntax defined in Section 3.
> **Step 2:** Write `main.py` that uses `watchdog` to monitor a folder called `./workspace`.
> **Step 3:** If a file is added to `./workspace`, trigger the interpreter to print the current 'Context' of that workspace based on the `.logos` file present.
> Use **Clean Architecture**. Ensure the code is modular."

---

