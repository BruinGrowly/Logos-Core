# Logos Core: The Semantic Overlay

**Version:** 0.5 (Bicameral Mind)
**Status:** Operational Prototype
**Architecture:** Hybrid Deterministic/Semantic

## 1. Project Overview

**Logos** is a Semantic Operating System overlay for Windows. It introduces a "Meaning Layer" between the user and the file system. Instead of managing files based solely on their location or extension, Logos manages them based on their *Context* and *Intent*.

The system now operates with a **Bicameral Mind**:
1.  **Reflex (Left Hemisphere):** Instant, rule-based execution (e.g., "If file contains 'invoice', move to Finance").
2.  **Intuition (Right Hemisphere):** Vector-based conceptual matching (e.g., "If file *feels* like a payment confirmation, move to Finance").

---

## 2. Architecture

```text
/logos-core
│
├── /cortex          (The Brain)
│   ├── interpreter.py     # Parses .logos manifests
│   ├── rules_engine.py    # Strict Logic (Reflex)
│   ├── semantic_engine.py # Fuzzy Logic (Intuition/Vectors)
│   └── actuator.py        # The Hands (File Ops, Logging)
│
├── /sensory         (The Senses)
│   └── watcher.py         # File System Monitor (Watchdog)
│
├── /memory          (The Hippocampus)
│   └── vector_memory.py   # Local Vector Store (all-MiniLM-L6-v2)
│
├── /workspace       (The Membrane)
│   └── project.logos      # The Configuration Manifest
│
├── main.py          # Entry Point
└── requirements.txt # Dependencies
```

---

## 3. Installation & Usage

### Prerequisites
*   Python 3.10+
*   Windows OS (tested on win32)

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/BruinGrowly/Logos-Core
    cd Logos-Core
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This includes `sentence-transformers` (~80MB model download on first run).*

3.  **Run the System:**
    ```bash
    python main.py
    ```

4.  **Interact:**
    *   The system monitors the `./workspace` folder.
    *   Drop files into this folder to trigger rules defined in `project.logos`.

---

## 4. Configuration: The `.logos` Syntax

The system is controlled by `project.logos` files placed in monitored directories.

### Example Configuration

```yaml
Manifest: Project_Alpha
Type: Entity

Rules:
  # --- REFLEX RULES (Strict/Fast) ---
  
  # Rule 1: Organization by Keyword
  - Trigger:
      Type: "File_Event"
      Condition: "Filename"
      Contains: ["invoice", "receipt", "bill"]
    Action:
      Type: "Move_File"
      Destination: "./Financial_Docs"

  # Rule 2: Security by Extension
  - Trigger:
      Type: "File_Event"
      Condition: "Extension"
      Is: [".secret", ".key"]
    Action:
      Type: "Log_Event"
      Message: "Secure file detected. Verify encryption."

  # --- INTUITION RULES (Fuzzy/Smart) ---

  # Rule 3: Semantic Matching
  # Matches files like "payment_doc_001.txt" even if they don't say "invoice"
  - Trigger:
      Type: "Semantic_Match"
      Concept: "Payment Confirmation"  # The system calculates the vector for this
      Threshold: 0.4                   # 0.0 to 1.0 similarity score
    Action:
      Type: "Move_File"
      Destination: "./Financial_Docs"
```

---

## 5. How It Works (The Pipeline)

When a file enters the workspace:

1.  **Sensation:** `watcher.py` detects the file creation/modification.
2.  **Interpretation:** `interpreter.py` reads the local `project.logos` to understand the rules.
3.  **Reflex Check:** `rules_engine.py` checks for strict matches (Filename/Extension).
    *   *If Match:* The **Actuator** executes the action immediately.
4.  **Intuition Check:** If no reflex matches, `semantic_engine.py` activates.
    *   It encodes the filename into a vector.
    *   It compares it against the "Concepts" defined in your rules.
    *   *If Similarity > Threshold:* The **Actuator** executes the action.

---

## 6. Development Roadmap

*   [x] **Phase 1:** Interpreter (YAML Parsing)
*   [x] **Phase 2:** Watcher (File System Events)
*   [x] **Phase 3:** Reflex Engine (Strict Logic)
*   [x] **Phase 4:** Actuator (File Operations)
*   [x] **Phase 5:** Semantic Brain (Local Vector Embeddings)
*   [ ] **Phase 6:** Deep Content Reading (OCR/PDF Parsing)
*   [ ] **Phase 7:** Feedback Loop (Reinforcement Learning)

---

**Logos Core** — *Code with Meaning.*