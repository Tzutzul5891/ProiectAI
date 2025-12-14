# SmarTest - Proiect IA (Streamlit)

SmarTest este o aplicație locală pentru generare de probleme tip examen și evaluare automată. Constrângere de design: **fără apeluri la API-uri LLM în runtime**; evaluarea și soluțiile sunt **locale/deterministe**.

## ✅ Implementat acum

- **Generare probleme (local):**
  - **Jocuri:** matrice 2x2 + detectare Echilibru Nash pur (`app/modules/games.py`).
  - **Căutare:** N-Queens, Turul Calului (5x5/6x6), Turnurile din Hanoi (3/4 tije, 3–5 discuri) (`app/modules/search.py`).
  - **CSP:** Graph Coloring (k-coloring) cu solver backtracking (`app/modules/graph_coloring.py`).
  - **CSP (Cerința 3):** CSP generic + solver Backtracking cu opțiuni MRV / Forward Checking / AC-3, pe instanțe JSON predefinite (`app/modules/csp.py`, `app/data/csp_instances/*.json`).
- **UI Streamlit (interactiv):**
  - tablă interactivă N-Queens, Turul Calului și Turnurile din Hanoi (`app/gui/components.py`).
- **Mod Test (multi-întrebări):**
  - selectezi capitole/subiecte + N întrebări, navighezi între ele și poți exporta PDF separat pentru test vs answer key (`main.py`, `app/utils/pdf_generator.py`).
- **Evaluare (local/determinist):**
  - Nash: scor semantic pe explicație + verificare coordonate (regex) (`app/evaluator/semantic.py`).
  - N-Queens: validare exactă a configurației + scor parțial.
  - Turul Calului: validare mișcări + scor euristic local.
  - Hanoi: validare corectitudine + eficiență față de optim.
  - Graph Coloring: 0–100% (validare + scor parțial pe conflicte).
- **Teorie (Cerința 1):** întrebări „Alegere Strategie” (strategie + justificare scurtă, scoring exact + parțial) (`app/modules/strategy_choice.py`, `app/evaluator/strategy_choice.py`).
- **Export PDF:** generare subiect PDF (`app/utils/pdf_generator.py`).

## 🧩 CSP: Backtracking cu FC/MRV/AC-3 (Cerința 3)

În modul acesta primești un CSP **predefinit** (din fișiere JSON) cu:
- variabile + domenii
- constrângeri (ex: `all_different`, constrângeri binare)
- asignare parțială
- metoda cerută (MRV / FC / AC-3)

Tu completezi **doar variabilele rămase**, iar aplicația calculează soluția determinist (BT + opțiunile cerute) și îți dă scor `0–100` pe potrivirea exactă per variabilă.

### Unde sunt instanțele

Instanțe: `app/data/csp_instances/*.json` (poți adăuga oricâte).

### Format JSON (minim)

```json
{
  "id": "exemplu_1",
  "variables": ["A", "B", "C"],
  "domains": { "A": [1,2,3], "B": [1,2,3], "C": [1,2,3] },
  "constraints": [
    { "type": "all_different", "vars": ["A","B","C"] },
    { "type": "less_than", "vars": ["A","B"] }
  ],
  "partial_assignment": { "B": 2 },
  "method": "MRV/FC/AC-3"
}
```

### Constrângeri suportate (în `constraints`)

- `all_different` (n-ary, se descompune în `!=` pentru AC-3)
- binare: `not_equal`, `equal`, `less_than`, `greater_than`
- binare numerice: `sum_equals`, `sum_not_equals`, `abs_diff_equals`, `abs_diff_not_equals`
- tabele: `allowed_pairs`, `forbidden_pairs`

### Opțiuni solver (în instanță)

- `method`: string sau listă (ex: `"MRV/FC/AC-3"` sau `["MRV","FC","AC-3"]`)
- opțional `ac3_mode`: `preprocess` / `interleaved` / `both` (pentru AC-3 ca preprocesare și/sau intercalat - MAC)

### Cum testezi în UI

- `streamlit run main.py`
- Mod: **O singură întrebare**
- Tip problemă: **`CSP (BT + FC/MRV/AC-3)`**
- Alege instanța din dropdown → **Încarcă instanța**
- Completezi în format `X=valoare, Y=valoare` → **Verifică Răspunsul**
- Opțional: descarci PDF-ul de subiect din stânga (**Descarcă Subiectul (PDF)**)

## 🧠 Teorie: „Alegere Strategie” (Cerința 1)

Aceasta este o întrebare de **teorie**: primești o problemă (N‑Queens / Hanoi generalizat / Graph Coloring / Knight’s Tour) + o **instanță** (dimensiune, k, nr. tije etc.), iar tu trebuie să:

1) **alegi o strategie** dintr-o listă fixă (dropdown în UI)  
2) scrii o **justificare scurtă** (2–3 propoziții)

Important: alegerea strategiei **nu pornește un algoritm** și **nu schimbă alte întrebări**. Te afectează doar prin **scorul** obținut la această întrebare.

**Cum se generează**
- Generatorul alege una dintre cele 4 familii de probleme + o instanță (`app/modules/strategy_choice.py`).
- Pentru fiecare familie există un „gold answer” (strategie + 2–3 motive standard).

**Cum se evaluează (fără LLM)**
- `100%` dacă strategia aleasă este exact cea corectă (match pe label).
- scor parțial dacă alegi o strategie „aproape” (ex: backtracking simplu vs backtracking cu MRV/Forward Checking).
- justificarea este verificată opțional pe cuvinte‑cheie (doar pentru feedback; nu schimbă scorul by default).

Evaluator: `app/evaluator/strategy_choice.py`

**Cum o testezi în UI**
- `streamlit run main.py`
- Mod: **Test (N întrebări)** → la **Subiecte** bifează `Alegere Strategie (Cerința 1)` → generează test.
- Completezi dropdown + justificare, apoi **Finalizează testul** ca să vezi scorul.
- Pentru a vedea răspunsul corect, bifează **„Arată și answer key în aplicație”** în ecranul de rezultate sau descarcă **Answer Key (PDF)**.

## 🔌 Convenție: ce returnează un generator

Convenția nouă (recomandată) este ca fiecare generator să întoarcă un `ProblemInstance` (vezi `app/modules/base_problem.py`):

- `data`: conținut structurat pentru UI/PDF (tablă, matrice etc.)
- `prompt`: enunțul problemei
- `solution`: soluția în format structurat (dacă există)
- `explanation`: explicația gold standard (text)
- `metadata`: câmpuri extra (dimensiuni, start_pos, număr mutări optime etc.)

Pentru compatibilitate cu UI-ul curent, clasele expun în continuare `generate_problem()` (legacy) care întoarce `(data, explanation)`.

## 🧱 Modele: `Question` & `TestSession`

- `Question` împachetează enunțul + datele + answer key (răspuns/expl.) într-un singur obiect: `app/models/test_session.py`.
- `TestSession` ține o listă de `Question` + index curent + răspunsuri/scoruri (pentru teste cu N întrebări).

Exemplu `Question` serializat (dict): `app/models/test_session.py` (`EXAMPLE_QUESTION_DICT`). Pentru PDF, poți folosi `Question.pdf_kwargs()` și `create_pdf(**kwargs)`.

```py
from app.models.test_session import EXAMPLE_QUESTION_DICT
from app.utils.pdf_generator import create_pdf

pdf_bytes = create_pdf(
    problem_type=EXAMPLE_QUESTION_DICT["metadata"]["ui_label"],
    requirement=EXAMPLE_QUESTION_DICT["prompt_text"],
    matrix_data=EXAMPLE_QUESTION_DICT["data"],
)
```

## 🛠️ Rulare

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## ⚙️ Opțiuni offline / determinism

- **Fără download de modele (offline strict):** `SMARTEST_LOCAL_MODELS_ONLY=1`
- **Model SBERT local (path):** `SMARTEST_SBERT_MODEL=/cale/către/model`
- **Generare reproductibilă (seed):** `SMARTEST_SEED=42`

Evaluatorul semantic încearcă SBERT local; dacă nu poate încărca modelul, folosește un fallback lexical determinist.

## 🗂️ Structură proiect

```
ProiectAI/
├── app/
│   ├── evaluator/      # evaluare (semantic, exact)
│   ├── gui/            # componente UI Streamlit
│   ├── modules/        # generatoare de probleme (search, games, viitor: csp/adversarial)
│   └── utils/          # utilitare (PDF, helpers)
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

## 🧭 Ce urmează

- Mutarea logicii de enunț/PDF pe `ProblemInstance.prompt` (mai puțin duplicat în `main.py`).
- Extindere instanțe CSP / constrângeri + implementări reale în `app/modules/adversarial.py`.
- `app/utils/pdf_parser.py`: parsare PDF -> structură internă (dacă e necesar).
- Teste minimale pentru generatoare/evaluatori (local, determinist).
