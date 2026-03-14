# BSL (British Sign Language) Data Sources Research

Research date: 2026-03-09

## Executive Summary

This report catalogues available BSL data sources for building a BSL sign language generation system. The most actionable resources are: BSL SignBank (~2,500 sign videos, open source Django app), BSLDict (14K video clips / 9K vocabulary from signbsl.com), and the BANZ-FS fingerspelling dataset (35K+ instances, CC BY 4.0). For grammar rules to drive English-to-BSL gloss translation, the English2BSL paper (ACL 2023) provides a rule-based framework. SignWriting data for BSL exists in SignPuddle but is sparse. The BOBSL dataset (1,400 hours) is large but has restrictive BBC licensing that prohibits commercial use.

---

## 1. BSL SignBank (UCL)

**URL:** https://bslsignbank.ucl.ac.uk/

### What it is
- First usage-based dictionary of BSL, built from the BSL Corpus
- Created at the Deafness, Cognition and Language Research Centre (DCAL), UCL
- Based on signs used by 249 Deaf people filmed across 8 UK cities

### Content
- **~2,500 lexical signs** (highly conventionalised signs in both form and meaning)
- Each entry has: annotation ID gloss, English keyword translations, phonological features (handshape, location), video demonstration
- Registered users get access to: Sign Number, ID gloss, Annotation ID Gloss, Phonology, Definitions and Notes
- Includes: Number Signs, Colour Signs, Country Signs, UK Place Name Signs, Finger Spelling

### Video format
- Videos are served through the Django web application
- Stored in a `GLOSS_VIDEO_DIRECTORY` within the Django media structure
- Format appears to be MP4 (the signbank-video module uses H.264 codec with no audio via FFmpeg)
- Individual sign videos are suitable as a lookup lexicon -- they show isolated signs

### Programmatic access
- **Source code is open source** (BSD-3-Clause license): https://github.com/Signbank/BSL-signbank
- Built in Django (Python 33%, JavaScript 39%, CSS 18%, HTML 9%)
- The signbank-video module (https://github.com/Signbank/signbank-video) handles video storage with a category/tag system
- **No documented public API** for downloading videos programmatically
- However, being open source Django, the video URL structure is predictable from the code (category + tag = gloss ID)
- The `signglossR` R package does NOT support BSL SignBank (only ASL and Swedish)
- **NOTE**: The signbank-video repo was archived January 2026 (read-only)

### License
- Free to browse publicly
- BSD-3-Clause for the software
- Content license not explicitly stated -- likely academic/research use
- Registration required for advanced features

### Feasibility assessment
- **HIGH** for building a lookup lexicon of ~2,500 common signs
- Videos are isolated signs, perfect for a concatenative sign generation system
- Open source codebase means video URLs can be reverse-engineered
- Limitation: only ~2,500 signs, covering core vocabulary but not comprehensive

---

## 2. BSL Corpus Project

**URL:** https://bslcorpusproject.org/

### What it is
- A collection of video clips showing Deaf people using BSL
- 249 deaf signers from 8 UK regions (London, Bristol, Cardiff, Birmingham, Newcastle, Manchester, Glasgow, Belfast)
- Funded by ESRC, 2008-2011
- Managed by DCAL at UCL

### Data access (via CAVA repository)
**URL:** https://bslcorpusproject.org/cava/

Two tiers:
1. **Open Access** (no login): Narratives and lexical elicitation videos -- viewable in browser, downloadable after agreeing to T&Cs
2. **Restricted Access** (requires End User License, may take weeks): Interviews and conversation data

### Annotations
- ELAN annotation files available for some data (Bristol, Birmingham, London, Manchester conversations; Belfast and Glasgow narratives)
- Lexical-level annotations with English translations
- Annotation conventions documented (v3.0, March 2017)

### Also available via
- UK Data Service ReShare: https://reshare.ukdataservice.ac.uk/851521/ (metadata only; actual data on CAVA)

### Feasibility assessment
- **MEDIUM** -- this is continuous signing data, NOT isolated sign clips
- Useful for training ML models or validating grammar rules
- Not directly suitable as a sign lookup lexicon
- ELAN annotations could be used to extract sign boundaries and create a mapping
- Licensing unclear for building a product

---

## 3. BOBSL (BBC-Oxford British Sign Language Dataset)

**URL:** https://www.robots.ox.ac.uk/~vgg/data/bobsl/

### What it is
- Large-scale dataset of BSL-interpreted BBC broadcast footage with English subtitles
- 1,940 episodes, ~1,400 hours
- 1.2 million sentences
- 37 distinct interpreters
- 25,000 manually annotated isolated sign instances

### Format
- Interpreter-cropped signing videos (MP4 and LMDB frame formats)
- Subtitle sentences and metadata
- Multiple annotation types: isolated signs, fingerspelling, continuous sequences, aligned subtitles
- Pre-computed video features (I3D, SWIN, OpenPose, RAFT models)

### License
- **RESTRICTIVE**: Requires BBC Terms of Use agreement
- **"Use is not permitted by independent researchers, students in secondary education, or commercial organisations"**
- Academic use only with institutional backing

### Feasibility assessment
- **LOW** for our purposes due to licensing restrictions
- Excellent resource for academic research but cannot be used commercially
- The 25K annotated isolated signs would be very useful if licensing permitted

---

## 4. BSLDict (from signbsl.com)

**URL:** https://github.com/gulvarol/bsldict

### What it is
- Research project for spotting signs in continuous video using dictionary examples
- Won Best Application Paper at ACCV 2020
- **14,000 video clips** for a vocabulary of **9,000 words and phrases**
- Videos sourced from signbsl.com (BSL sign aggregation platform)

### Content
- Dictionary entries include multiple video examples per word
- Shows regional variations
- Isolated signs performed at slower rates for clarity
- Videos show still signer pausing at start and end (clean boundaries)

### Format
- MP4 video clips
- Downloadable via the research project's download scripts (uses wget and youtube-dl)

### License
- Users must "check the license before downloading or using the BSLDict dataset"
- Videos are sourced from signbsl.com which aggregates from multiple contributors
- **License status unclear -- need to verify with signbsl.com**

### signbsl.com itself
- **URL:** https://www.signbsl.com/
- Largest collection of BSL video signs online (20,000+ videos in their app)
- Aggregates videos from multiple sources
- Has a page at `/sign/application-program-interface-api` suggesting some API exists
- The signbsl.com data was "made available by Daniel Mitchell" for the BSLDict project

### Feasibility assessment
- **HIGH** potential if licensing can be confirmed
- 9K vocabulary with 14K clips is substantial
- Videos are isolated signs, perfect for lookup
- Need to contact Daniel Mitchell / signbsl.com about commercial licensing

---

## 5. BSL Fingerspelling Data Sources

### 5.1 BANZ-FS Dataset (BANZSL Fingerspelling)

**URL:** https://openreview.net/forum?id=GMR9BUsPbq

- **35,000+ video-aligned fingerspelling instances**
- Covers BSL, Auslan, and NZSL (all use two-handed fingerspelling)
- Three sources: news broadcast interpretation, lab recordings, online vlogs
- Multi-level annotation: video <-> subtitles, video <-> fingerspelled letters, video <-> target lexicons
- **License: CC BY 4.0** (Creative Commons Attribution)
- ICLR 2026 submission
- Addresses two-handed fingerspelling specifically (handshape coarticulation, self-occlusion, rapid transitions)
- **FEASIBILITY: HIGH** -- CC BY 4.0 is permissive, covers BSL two-handed fingerspelling

### 5.2 Kaggle: BSL Numbers & Alphabet Hand Position for MediaPipe

**URL:** https://www.kaggle.com/datasets/erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe

- 34,000 images of BSL numbers and alphabet
- Numbers 0-10 and alphabet letters (excluding H, J, Y)
- 21 hand landmark locations from MediaPipe (2 CSV files)
- 2 pre-trained SVM models included
- **License: Unknown**
- **LIMITATION**: Only single-hand landmarks (21 points = one hand), excludes 3 letters
- BSL uses TWO-handed fingerspelling, so single-hand tracking is incomplete
- **FEASIBILITY: LOW** -- incomplete coverage, unknown license

### 5.3 BSL Fingerspelling Alphabet Charts

**URL:** https://www.british-sign.co.uk/fingerspelling-alphabet-charts/

- Printable PDFs of the two-handed BSL alphabet (right-handed and left-handed versions)
- Static images only, no video
- License not specified
- Could be used as reference material for generating fingerspelling overlays
- **FEASIBILITY: MEDIUM** -- useful as reference but not as training data

### 5.4 BSL Fingerspelling Detection & Recognition (GitHub)

**URL:** https://github.com/yuankit/BSLfingerspelling-Detection-and-Recognition

- Research code for BSL fingerspelling detection
- Related to BOBSL dataset annotations
- **FEASIBILITY: LOW** -- tied to BOBSL licensing restrictions

---

## 6. BSL Grammar Resources (for English-to-BSL Gloss Translation)

### 6.1 Core Grammar Rules

Based on multiple sources, here are the key BSL grammar rules that could be encoded into a Bedrock prompt for English-to-BSL gloss translation:

#### Word Order: TIME - TOPIC - COMMENT
BSL uses a Topic-Comment structure, not English's Subject-Verb-Object:
- English SVO: "I eat an apple"
- BSL gloss: `APPLE I EAT` (Object-Subject-Verb)
- Full structure: **Time frame -> Location -> Object -> Subject -> Verb -> Question word**

Example:
- English: "Why was the black cat climbing the tree in your garden yesterday?"
- BSL gloss: `YESTERDAY YOUR GARDEN TREE BLACK CAT CLIMB WHY`

#### Time References
- Time is always established FIRST in a BSL sentence
- BSL verbs do NOT change form for tense (no past/present/future conjugation)
- The time frame at the start sets the tense for everything that follows
- Examples: `YESTERDAY`, `TOMORROW`, `LAST-WEEK`, `BEFORE`, `NOW`

#### Questions
- **Yes/No questions**: Raised eyebrows, forward head lean
- **WH-questions** (who, what, where, when, why, how): Furrowed/lowered eyebrows
- Question words go at the END of the sentence:
  - English: "What is your name?" -> BSL: `NAME YOU WHAT`
  - English: "Where do you live?" -> BSL: `LIVE YOU WHERE`

#### Negation
- Expressed through: headshake, negative facial expression, lip pattern
- Negative headshake accompanies the sign being negated
- Dedicated negative signs: `NOT`, `NEVER`, `NOTHING`, `NOBODY`
- No auxiliary verb constructions (no "don't", "doesn't", "won't")

#### Articles and Function Words
- Remove: a, an, the, is, are, am, was, were, be, been, being
- Remove: to (as infinitive marker), of, do, does, did
- BSL has no articles or copula verbs
- Prepositions are often replaced by spatial positioning/classifiers

#### Pronouns
- Handled through spatial referencing (pointing to locations in sign space)
- Index pointing replaces he/she/it/they
- Possessives use flat-hand pointing

#### Non-Manual Features (NMFs)
- Facial expressions are GRAMMATICAL (not just emotional)
- Raised eyebrows = yes/no question
- Furrowed eyebrows = WH-question
- Headshake = negation
- Puffed cheeks = large/lots
- Pursed lips = small/thin
- Mouth patterns often mirror English word beginnings
- Head tilt/nod for emphasis

#### Spatial Grammar
- Signers establish referents in 3D space
- Directional verbs move between established locations
- Classifier handshapes represent object categories and movements
- Role shifting (body shift) for reported speech/different perspectives

#### BSL vs. Signed Supported English (SSE)
- SSE follows English word order and just replaces words with signs
- BSL has its own grammar, word order, and linguistic structure
- BSL is a natural language; SSE is a manually coded representation of English
- A system should produce BSL gloss, not SSE, for authenticity

### 6.2 English2BSL: Rule-Based Translation Paper

**Citation:** Pinney, P.A. & Batista-Navarro, R. (2023). "English2BSL: A Rule-Based System for Translating English into British Sign Language." Proceedings of the Third Workshop on Language Technology for Equality, Diversity and Inclusion (LTEDI), ACL.

**URL:** https://aclanthology.org/2023.ltedi-1.2/

**Key approach:**
- POS tagging using Penn Treebank tagset
- Handcrafted rules for reordering English to BSL gloss order
- Rules include:
  - SVO to SOV conversion
  - Time expressions moved to sentence-initial position
  - Articles and prepositions removed
  - Negation markers repositioned
  - Question words moved to sentence-final position
  - Pronouns handled through spatial referencing notation
  - Verb agreement marking and aspect
  - Classifier predicates for spatial representation

**FEASIBILITY: HIGH** -- this paper provides a concrete, implementable rule set

### 6.3 Signapse AI Approach

**URL:** https://www.signapse.ai/post/what-is-our-planned-approach-to-the-grammar-of-sign-language

- Commercial BSL translation company
- Uses LLM technology trained on English text + BSL gloss data
- Phased approach from SSE-like output toward full BSL grammar
- Incorporating topic-comment structure, directional verbs, NMFs
- Collaborating with Deaf interpreters and linguists
- Target: expressive signing with paragraph-level context by end of 2025

### 6.4 BSL Syntax Project

**URL:** https://bslcorpusproject.org/projects/bsl-syntax-project/

- First large-scale investigation of BSL grammatical system
- Combines experimental + spontaneous conversation data
- Corpus-based, cognitive/functional, sociolinguistic approaches
- **Reference grammar planned for 2026** at https://bslsignbank.ucl.ac.uk/about/grammar/
- When published, this will be the definitive BSL grammar reference

### 6.5 Other Grammar Resources

| Resource | URL | Notes |
|----------|-----|-------|
| BSL Grammar overview (knowledge.deck.no) | https://knowledge.deck.no/languages-and-linguistics/sign-languages/british-sign-language-bsl/bsl-grammar | General overview |
| Gareth Jones BSL word ordering | https://garethejones.wordpress.com/2008/03/21/bsl-word-ordering/ | Practical examples |
| Simple BSL grammar intro (appa.me.uk) | https://appa.me.uk/2017/a-simple-intro-to-bsl-grammar/ | Time-Topic-Action structure |
| BSL sentence structure forum | https://www.signlanguageforum.com/bsl/topic/22-sentence-structure-in-bsl/ | Community examples with gloss |
| Level 3 BSL grammar handout | http://www.deafsupport.org.uk/grammar-l3.pdf | Detailed rules (SSL cert issues) |
| Word order (BSL Wiki) | https://bsl.fandom.com/wiki/Word_order | Concise rules |

---

## 7. BSL Gloss Dictionaries / Word Lists

### 7.1 BSL SignBank Gloss List
- ~2,500 annotation ID glosses mapping English keywords to BSL signs
- Each sign has a unique Annotation ID Gloss (the canonical written form)
- Available through the BSL SignBank web interface
- Could potentially be extracted from the open source Django database

### 7.2 Scottish Sensory Centre BSL Glossary

**URL:** https://www.ssc.education.ed.ac.uk/BSL/

- ~2,000 signs across STEM subjects + Geography, Computing, Cyber Security, Data Science, Environmental Science, Astronomy, Marine Species
- Each entry: sign video, fingerspelled form, BSL explanation video with English translation
- A-Z index for browsing
- **License: "All SSC BSL Glossary videos are Intellectual property of University of Edinburgh and cannot be used elsewhere without express permission"**
- **FEASIBILITY: LOW** for re-use due to restrictive IP claim; useful as reference

### 7.3 Signature BSL Glossary

**URL:** https://www.signature.org.uk/bsl-glossary-2/

- Professional awarding body for BSL qualifications
- Online glossary resource
- Not assessed for programmatic access

### 7.4 signbsl.com Dictionary

**URL:** https://www.signbsl.com/

- 20,000+ videos in their app
- Largest online BSL video collection
- Aggregates from multiple sources
- Has potential API endpoint
- **FEASIBILITY: MEDIUM-HIGH** -- large vocabulary but licensing unclear

---

## 8. SignWriting Data for BSL

### 8.1 SignPuddle GB (BSL SignWriting Dictionary)

**URL:** https://www.signbank.org/signpuddle/index4.html (select "SignPuddle GB")

- BSL dictionary exists in SignPuddle using Sutton SignWriting notation
- Data stored as SQLite 3 databases, downloadable individually
- ISO code: `sgn-GB`
- SignPuddle 3 API exists at https://signpuddle.com/client/api/dictionary.html
- API endpoints include: search by sign, search by text terms, search by SignWriting
- **Entry count unknown** -- likely small (SignWriting adoption for BSL is limited)
- BSL has no widely accepted writing system; notation is primarily done via glossing

### 8.2 SignWriting BSL Dictionary PDF

**URL:** https://signwriting.org/archive/docs14/sw1384_Dictionary_British_Sign_Language_2023_0924.pdf

- PDF dictionary of BSL in SignWriting format (1.7MB, dated September 2023)
- Could not be read programmatically in this research session
- Likely contains a modest number of BSL signs in SignWriting notation

### 8.3 SignWriting Server

**URL:** https://swserver.wmflabs.org/

- Provides SVG images for individual SignWriting symbols
- Completed 2D signs in Formal SignWriting
- Data from SignPuddle Online available as SQLite 3 databases
- API documented using API Blueprint
- Responds with SVG or JSON data

### Feasibility assessment
- **LOW** for BSL specifically -- SignWriting adoption for BSL is minimal
- BSL community primarily uses glossing (English words in CAPS) rather than SignWriting
- The SignPuddle BSL dictionary likely has far fewer entries than ASL or other sign languages
- Could be useful as supplementary data but not as a primary resource

---

## 9. Other Notable Resources

### 9.1 Sign Language Processing Research Hub

**URL:** https://research.sign.mt/

- Comprehensive survey of sign language processing tasks
- Lists datasets, tools, avatar systems (JASigning, PAULA, SiMAX)
- Covers text-to-pose and pose-to-video generation
- References BOBSL and other BSL resources

### 9.2 GenASL (AWS)

**URL:** https://aws.amazon.com/blogs/machine-learning/genasl-generative-ai-powered-american-sign-language-avatars/

- AWS blog on generative AI-powered ASL avatars
- Not BSL-specific but demonstrates the technical approach
- Could inform the architecture of a BSL system on AWS

### 9.3 SignAvatar

**URL:** https://www.signavatar.org/

- Generate sign language from speech using AI
- General sign language production research

### 9.4 Text2Sign (Springer)

**URL:** https://link.springer.com/article/10.1007/s11263-019-01281-2

- Academic paper on sign language production using NMT and GANs
- Covers the text -> gloss -> sign pipeline

---

## 10. Recommended Architecture for a BSL Generation System

Based on this research, the most feasible approach would be:

### Step 1: English to BSL Gloss (using Bedrock/LLM)
- Use the grammar rules from section 6 to create a Bedrock prompt
- Key rules to encode: TIME-TOPIC-COMMENT order, remove articles/copula, move question words to end, establish negation via dedicated signs
- Reference the English2BSL paper for the specific POS-based reordering rules
- The LLM approach (as used by Signapse) may be more robust than pure rules

### Step 2: BSL Gloss to Sign Video Lookup
- Primary source: BSLDict/signbsl.com (~9K vocabulary, 14K clips) -- if licensing permits
- Secondary source: BSL SignBank (~2,500 signs) -- open source, reliable
- Fallback: fingerspelling for unknown words using BANZ-FS reference data

### Step 3: Fingerspelling for Unknown Words
- Use the two-handed BSL alphabet
- Reference: BANZ-FS dataset (CC BY 4.0) for training/validation
- Static reference: british-sign.co.uk alphabet charts

### Key Gaps
1. **No single comprehensive, openly-licensed BSL video dictionary** exists
2. **Non-manual features** (facial expressions) cannot be conveyed through simple video concatenation
3. **Spatial grammar** (directional verbs, classifier predicates) requires generative approaches, not lookup
4. **BSL grammar reference** is still being written (planned 2026)

---

## Summary Table

| Resource | Signs/Size | License | Video? | Programmatic? | Feasibility |
|----------|-----------|---------|--------|---------------|-------------|
| BSL SignBank (UCL) | ~2,500 | BSD-3 (software) | Yes (MP4) | Via source code | HIGH |
| BSLDict/signbsl.com | 9K vocab, 14K clips | Unclear | Yes (MP4) | Download scripts | HIGH (if licensed) |
| BSL Corpus (CAVA) | 249 signers, continuous | Research use | Yes | After registration | MEDIUM |
| BOBSL (BBC) | 1,400 hours, 25K annotated | BBC restrictive | Yes | After agreement | LOW (licensing) |
| BANZ-FS (fingerspelling) | 35K+ instances | CC BY 4.0 | Yes | TBD | HIGH |
| Kaggle BSL alphabet | 34K images | Unknown | No (images) | Kaggle API | LOW (incomplete) |
| SSC BSL Glossary | ~2,000 STEM terms | Univ. Edinburgh IP | Yes | No | LOW (restrictive) |
| SignPuddle BSL | Unknown (likely small) | Open | No (SignWriting) | SQLite/API | LOW |
| English2BSL rules | N/A | Academic paper | N/A | N/A | HIGH (for grammar) |
