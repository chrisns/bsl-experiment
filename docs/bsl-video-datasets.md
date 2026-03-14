# Comprehensive BSL Video Dataset Research

Research date: 2026-03-12

## Executive Summary

This report covers all identified sources of BSL (British Sign Language) video data for ML sign recognition training. Sources are categorised into: Academic Datasets, Online BSL Dictionaries (scrapeable), Educational Glossaries, Multilingual/Cross-language Datasets, and Broadcast/Interpreted Content. The most promising **new** sources beyond those already known are: the Dicta-Sign BSL lexicon (~1,000 concepts with video), the Scottish Sensory Centre STEM glossaries (~2,000+ terms with video), YouTube-ASL methodology (applicable to BSL YouTube content), and the BANZ-FS fingerspelling dataset (35K+ instances, CC BY 4.0).

---

## TIER 1: Large-Scale Academic BSL Datasets

### 1. BSL-1K (Oxford VGG, 2020)

- **URL:** https://www.robots.ox.ac.uk/~vgg/research/bsl1k/
- **Paper:** arXiv:2007.12131
- **Size:** 1,000 sign vocabulary, 1,000 hours of video
- **Signers:** Multiple (from broadcast footage)
- **Access:** Subsumed into BOBSL (see below). BSL-1K itself is not separately downloadable; users are directed to BOBSL.
- **Format:** Broadcast video clips with automatic annotations via mouthing cue detection
- **License:** Same as BOBSL (BBC Terms of Use)
- **How to get it:** Apply via BBC R&D website (see BOBSL entry)
- **Key innovation:** Uses mouthing cues (lip movements that mirror spoken words) to automatically annotate signs in broadcast footage. This methodology could be applied to any BSL broadcast content.

### 2. BOBSL (BBC-Oxford British Sign Language Dataset, 2021)

- **URL:** https://www.robots.ox.ac.uk/~vgg/data/bobsl/
- **Paper:** arXiv:2111.03635
- **Size:** 1,940 episodes, ~1,400 hours, 1.2M sentences, 77,000-word vocabulary
- **Signers:** 37 distinct interpreters
- **Annotations:** 25K isolated sign instances (manual), 32K aligned subtitles, 5K fingerspelling instances, 48K continuous sign glosses, 6.1M background captions (automatic)
- **Access:** Requires BBC Terms of Use agreement. Academic/research institutions only. **Not available to independent researchers, students, or commercial organisations.**
- **Application URL:** https://www.bbc.co.uk/rd/projects/extol-dataset
- **Format:**
  - Interpreter-cropped MP4 videos (331 GB) or LMDB frames (1.5 TB)
  - Pre-computed features: I3D (54 GB), SWIN (175-262 GB), keypoints (675 GB), optical flow (761 GB)
  - Subtitle files, metadata
- **License:** BBC Terms of Use (restrictive, non-commercial)
- **How to get it:** Submit agreement via BBC R&D website, receive personal password upon approval

### 3. BSLDict (Oxford VGG / signbsl.com, 2020)

- **URL:** https://github.com/gulvarol/bsldict
- **Paper:** arXiv:2010.04002 (ACCV 2020 Best Application Paper)
- **Size:** ~14,000 video clips, ~9,000 words/phrases
- **Signers:** 150+ different signers (aggregated from signbsl.com contributors)
- **Access:** Open download via provided scripts
- **Format:** MP4 video clips, isolated signs with clean start/end pauses
- **License:** Must check license before use. Videos sourced from signbsl.com (multiple original contributors).
- **How to get it:**
  ```bash
  git clone https://github.com/gulvarol/bsldict
  cd bsldict && bash download_bsldict_metadata.sh  # 148MB metadata
  cd models && bash download_models.sh              # 118MB pretrained models
  # Full videos downloadable via separate script
  ```
- **Notes:** Already in use by this project (944-sign vocabulary model trained from BSLDict videos). Videos are sourced from signbsl.com which aggregates content from University of Bristol, Deaf Studies Trust, NDCS, Exeter Deaf Academy, and many individual contributors.

### 4. BSL Corpus Project / CAVA (UCL, 2008-2011)

- **URL:** https://bslcorpusproject.org/cava/
- **Size:** 249 deaf signers from 8 UK regions (London, Bristol, Cardiff, Birmingham, Newcastle, Manchester, Glasgow, Belfast)
- **Content:** Narratives, lexical elicitation, interviews, conversations
- **Signers:** 249 (diverse ages, regions, backgrounds)
- **Access:**
  - **Open Access** (no login): Narratives and lexical elicitation data
  - **Restricted Access** (End User License, weeks to process): Interviews and conversations
- **Format:** Multi-camera video (3 angles per activity), ELAN annotation files (.eaf), English translations
- **Annotations:** Lexical-level annotations for conversation data from 4 cities, English translations for all narratives
- **License:** Open Access tier requires agreeing to T&Cs; Restricted Access requires End User License
- **How to get it:** Visit https://bslcorpusproject.org/cava/, agree to terms for open data, apply for license for restricted data
- **Also available via:** UK Data Service ReShare: https://reshare.ukdataservice.ac.uk/851521/

### 5. BSL SignBank (UCL/DCAL)

- **URL:** https://bslsignbank.ucl.ac.uk/
- **Source code:** https://github.com/Signbank/BSL-signbank (BSD-3-Clause)
- **Size:** ~2,500 lexical signs, 345+ entries for letter A alone (suggesting ~5,000-8,000 total entries including variants)
- **Signers:** Derived from BSL Corpus (249 signers)
- **Access:** Free public browsing. Registration required for advanced features. Individual login required for detailed gloss pages.
- **Format:** MP4 videos, phonological feature annotations (handshape, location), English keywords
- **URL pattern:** `/dictionary/words/[gloss]-1.html` (e.g., `/dictionary/words/abandon-1.html`)
- **License:** BSD-3-Clause for software; content license not explicitly stated
- **How to get it:** Browse web interface; source code is open for URL pattern analysis. Includes Number Signs, Colour Signs, Country Signs, UK Place Name Signs, Fingerspelling.

### 6. BANZ-FS Fingerspelling Dataset (ICLR 2026)

- **URL:** https://openreview.net/forum?id=GMR9BUsPbq
- **Size:** 35,000+ video-aligned fingerspelling instances
- **Languages:** BSL, Auslan (Australian), NZSL (New Zealand) -- all use two-handed fingerspelling
- **Sources:** News broadcast interpretation, lab recordings, online vlogs
- **Annotations:** Multi-level: video-subtitle alignment, video-letter alignment, video-lexicon alignment
- **License:** CC BY 4.0 (permissive)
- **Format:** Video clips with dense annotations
- **How to get it:** Links in the ICLR paper (CC BY 4.0 license)

---

## TIER 2: Online BSL Video Dictionaries (Scrapeable)

### 7. SignBSL.com -- LARGEST ONLINE BSL COLLECTION

- **URL:** https://www.signbsl.com/
- **Size:** 21,000+ videos, 150+ different signers, "thousands of words and phrases"
- **Creator:** Daniel Mitchell (2013-present)
- **Organisation:** Alphabetical browsing (A-Z), search, by category
- **URL pattern:** `/sign/[word-or-phrase]` (e.g., `/sign/hello`, `/sign/abandon`)
- **Video sources per entry:** Multiple variants from different organisations:
  - University of Bristol & Deaf Studies Trust
  - National Deaf Children's Society (NDCS)
  - Deafway
  - Exeter Deaf Academy
  - Evelina London
  - Famlingo
  - Karl O'Keeffe, Jayne Fletcher, and other individual signers
  - Scottish Sensory Centre
  - BSL Parlour (Nathanael Farley)
  - Deafax
  - SignMedia SMART
- **Video format:** HTML5 video player; video source URLs are loaded dynamically via JavaScript (not in static HTML). Would require browser automation (Playwright/Selenium) or network traffic analysis to extract video URLs.
- **Regional variants:** Different signs shown for the same word from different regions/sources
- **Metadata per video:** Signer/organisation attribution, source website link
- **Terms:** Copyright 2013-2026 by site operator. Privacy policy exists. No explicit open license.
- **Sister site:** SignASL.org (American Sign Language)
- **Feasibility:** HIGH for scraping -- consistent URL structure, large vocabulary, multiple signers per sign. Need to reverse-engineer JavaScript video loading. Contact Daniel Mitchell about licensing.

### 8. british-sign.co.uk BSL Dictionary

- **URL:** https://www.british-sign.co.uk/british-sign-language/dictionary/
- **Size:** 485 signs
- **Format:** Still images (NOT video), showing hand positions
- **Features:** Customisable background colour, left/right hand versions
- **Browsing:** Alphabetical, paginated (20 per page)
- **Terms:** Copyright british-sign.co.uk
- **Feasibility:** LOW -- only 485 signs, images not video

---

## TIER 3: Educational BSL Glossaries with Video

### 9. Scottish Sensory Centre BSL STEM Glossaries (University of Edinburgh)

- **URL:** https://www.ssc.education.ed.ac.uk/BSL/
- **Size:** Estimated 2,000+ signs across 11 subject areas:
  - Astronomy
  - Biology (~500+ terms across 26 categories)
  - Chemistry (~200+ terms)
  - Computing Science
  - Cyber Security
  - Data Science
  - Environmental Science
  - Geography
  - Marine Species (~100 animals in "Above and Below")
  - Mathematics
  - Physics (~500+ terms)
- **Format:** Video for each sign (sign demonstration, fingerspelled form, BSL explanation with English translation)
- **URL pattern:** `/BSL/[subject]/[term].html#start` (e.g., `/BSL/chemistry/acid.html#start`)
- **Browsing:** By subject category, or A-Z index
- **Signers:** University of Edinburgh staff/collaborators
- **Established:** 2007, actively expanding
- **License:** "All SSC BSL Glossary videos are Intellectual property of University of Edinburgh and cannot be used elsewhere without express permission"
- **How to get it:** Browsable online, systematic scraping possible via predictable URL structure. Would need permission from University of Edinburgh.
- **Feasibility:** MEDIUM -- substantial STEM vocabulary, structured and labelled, but restrictive IP claim. Good for expanding domain-specific vocabulary if permission obtained.

### 10. University of Surrey BSL Learning Resource

- **URL:** https://bsl.surrey.ac.uk/
- **Partner:** Dot Sign Language (nonprofit)
- **Content:** Structured BSL lessons across 4 subject areas:
  - Principles (deaf culture, fingerspelling, handshapes, non-manual features)
  - Meeting People (introductions, greetings, personal info)
  - Numbers & Dates (numerals, time, calendars, money)
  - Weather, Transport & Directions
- **Format:** Videos, interactive exercises, webcam practice
- **Size:** Limited vocabulary (educational resource, not dictionary)
- **Feasibility:** LOW -- small vocabulary, educational focus, terms unclear

### 11. Signature BSL Glossary

- **URL:** https://www.signature.org.uk/bsl-glossary-2/
- **Description:** Professional awarding body for BSL qualifications
- **Content:** Linguistic terminology definitions, GCSE consultation terms with BSL video (YouTube)
- **Size:** Unknown, focused on metalinguistic terms
- **Feasibility:** LOW -- terminology glossary, not a sign dictionary

---

## TIER 4: Multilingual Sign Language Datasets (Cross-language Transfer Potential)

### 12. Dicta-Sign (EU Project, 2009-2012) -- INCLUDES BSL

- **URL:** https://www.sign-lang.uni-hamburg.de/dicta-sign/
- **Lexicon URL:** https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_eng.html
- **Languages:** BSL, DGS (German), GSL (Greek), LSF (French)
- **Size:** ~1,000 concepts with video for each of 4 sign languages
- **Additional training data:** ~1,000 isolated signs with 10 identical + 10 varied copies per sign (at least for DGS/GSL)
- **Corpus:** 14+ informants per language, ~2 hours per session
- **URL pattern:** `/cs/cs_[number].html` (e.g., `cs_2.html` for "abroad")
- **Access:** Open browsing for concept lexicon. Corpus data has restricted-access area.
- **Contact:** thomas.hanke@sign-lang.uni-hamburg.de
- **Feasibility:** MEDIUM-HIGH -- ~1,000 BSL signs with video, structured and labelled. Good for expanding vocabulary with multilingual alignment. Need to verify video downloadability and terms.
- **Bonus:** Cross-language alignment enables transfer learning between BSL and other European sign languages.

### 13. YouTube-ASL (Google, 2023) -- METHODOLOGY APPLICABLE TO BSL

- **Paper:** arXiv:2306.15162
- **Size:** ~1,000 hours, 2,500+ unique signers (ASL only)
- **Methodology:** Scraping YouTube videos with English captions that contain sign language
- **BSL applicability:** The same pipeline could be applied to BSL YouTube content. Key requirements:
  - Identify BSL YouTube channels/videos
  - Match with English captions/subtitles
  - Filter for quality and sign language presence
- **License:** Research use (Google)
- **Feasibility for BSL:** MEDIUM -- methodology is replicable but requires significant engineering to build a BSL-specific YouTube pipeline

### 14. YouTube-SL-25 (2025)

- **Referenced in:** arXiv:2512.08040
- **Description:** Large-scale multilingual YouTube sign language dataset covering 25 sign languages
- **BSL inclusion:** Not confirmed in abstracts reviewed, appears focused on DGS/ASL
- **Access:** Via the paper authors
- **Feasibility:** MEDIUM -- if BSL is included, could be a significant resource

### 15. TikTok-SL-8 (2025)

- **Referenced in:** arXiv:2510.25413
- **Description:** TikTok videos across 8 sign languages
- **BSL inclusion:** Not confirmed
- **Access:** Via the paper authors
- **Feasibility:** LOW-MEDIUM -- unclear if BSL is included

### 16. SpreadTheSign

- **URL:** https://spreadthesign.com/en.gb/search/
- **Size:** 400,000+ signs globally across many sign languages
- **BSL content:** Yes, BSL is one of the supported languages
- **Terms:** "All content (videos, photos, illustrations, texts) is only for personal use." Commercial/research requires permission and licence from the European Sign Language Center.
- **Contact:** European Sign Language Center (nonprofit)
- **Feasibility:** MEDIUM -- large vocabulary but restrictive licensing. Contact required for any training data use.

### 17. DGS-Korpus (German Sign Language Corpus)

- **URL:** https://www.sign-lang.uni-hamburg.de/meinedgs/
- **Language:** German Sign Language (DGS)
- **Size:** Extensive (multiple cities, age groups)
- **Format:** MPEG4 video, iLex/ELAN annotations, SRT subtitles, OpenPose JSON, CMDI metadata
- **BSL relevance:** Not BSL, but useful for:
  - Transfer learning (European sign language)
  - Methodology reference for corpus building
  - OpenPose/keypoint extraction pipeline
- **Feasibility for BSL:** LOW (wrong language) but HIGH for methodology

### 18. sign-language-processing/datasets (GitHub)

- **URL:** https://github.com/sign-language-processing/datasets
- **Description:** Unified TensorFlow Datasets loader for sign language datasets
- **BSL entry:** `bsl_corpus` (v3.0) -- loads BSL Corpus data but "no videos/poses" according to README
- **Other datasets:** 20+ sign language datasets (ASL, DGS, Turkish, Chinese, etc.)
- **Feasibility:** LOW for BSL directly (no video in the BSL loader), but useful infrastructure

---

## TIER 5: Broadcast & Interpreted BSL Content

### 19. Lumo TV (formerly BSL Zone)

- **URL:** https://lumotv.co.uk/
- **Description:** World's leading deaf-led streaming platform, all content in BSL
- **Content types:** Drama, entertainment, factual, children's, comedy, sport, Deaflympics
- **Size:** Dozens of shows across multiple genres
- **BSL content:** All content features BSL
- **Structured vocabulary:** No dictionary or structured sign content -- entertainment programming only
- **Feasibility:** LOW for isolated sign training (no labelled vocabulary). MEDIUM for continuous sign recognition training if licensed.

### 20. BSL Bible Project

- **URL:** https://www.bslbible.org.uk
- **Content:** Gospel of Mark translated into BSL, chapter by chapter
- **Format:** YouTube-hosted videos
- **Organisation:** Highly structured by book, chapter, verse ranges with descriptive titles
- **Size:** 68+ passage sections for Mark 1-16
- **Feasibility:** LOW for sign recognition (continuous interpreted content, not isolated signs)

### 21. SignHealth BSL Health Video Library

- **URL:** https://www.signhealth.org.uk/health-video-library/
- **Description:** "Largest collection of health information in BSL"
- **Content:** 100+ health topics (cancer, depression, first aid, abuse, sexual health, vaccines, etc.)
- **Format:** BSL videos with English subtitles, hosted on YouTube/Vimeo
- **Organisation:** Categorised by topic, searchable
- **Feasibility:** LOW for isolated sign training, MEDIUM for domain-specific continuous sign data

### 22. British Deaf Association (BDA) BSL Videos

- **URL:** https://www.bda.org.uk
- **Content:** BSL information videos on health, wellbeing, hate crime, PIP, access to justice
- **Projects:** Family Sign at Home, Health & Wellbeing, Signed Voices
- **Size:** Multiple project-specific video series
- **Feasibility:** LOW (continuous BSL content, not labelled individual signs)

### 23. NHS BSL Videos

- **URL:** Various NHS pages
- **Content:** Health information in BSL with English subtitles
- **Feasibility:** LOW for isolated signs but the subtitle alignment could enable automatic annotation

---

## TIER 6: YouTube BSL Channels (Potential Scraping Sources)

The following YouTube channels produce structured BSL content that could potentially be scraped and processed:

### Known BSL YouTube content producers:

- **BSL Zone / Lumo TV** -- entertainment in BSL
- **NDCS (National Deaf Children's Society)** -- educational BSL videos
- **Exeter Deaf Academy** -- BSL educational content
- **University of Bristol / Deaf Studies Trust** -- academic BSL content
- **Karl O'Keeffe** -- individual BSL content creator
- **Jayne Fletcher** -- individual BSL content creator
- **BSL Parlour (Nathanael Farley)** -- BSL teaching content

### YouTube-BSL Pipeline Approach

Following the YouTube-ASL methodology (arXiv:2306.15162):
1. Identify BSL YouTube channels and videos
2. Download videos with English captions/subtitles
3. Use sign detection models to segment signing from non-signing
4. Use mouthing cue detection (BSL-1K methodology) to identify individual signs
5. Build aligned sign-text pairs

This is a significant engineering effort but could yield a large dataset.

---

## TIER 7: Related Resources

### 24. Dicta-Sign BSL Training Data

The Dicta-Sign project produced ~1,000 BSL isolated signs with multiple recordings per sign for the training data component. This is separate from the concept lexicon and could contain 10,000+ individual recordings. Contact: thomas.hanke@sign-lang.uni-hamburg.de

### 25. Irish Sign Language (ISL) Resources

ISL shares historical roots with BSL (both descended from Old BSL). ISL dictionaries could provide:
- Cross-reference data for sign similarity analysis
- Transfer learning opportunities
- Key resource: University of Galway ISL Dictionary

### 26. Kaggle BSL Numbers & Alphabet

- **URL:** https://www.kaggle.com/datasets/erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe
- **Size:** 34,000 images, numbers 0-10 and alphabet (excluding H, J, Y)
- **Format:** 21 hand landmark positions from MediaPipe (CSV)
- **License:** Unknown
- **Limitation:** Single-hand landmarks only; BSL uses two-handed fingerspelling
- **Feasibility:** LOW

### 27. Sign Language Processing Tools

- **OpenHands:** https://github.com/sign-language-processing/openhands -- language-agnostic framework
- **MediaPipe Holistic:** Google's hand/face/body pose estimation
- **OpenPose:** Body keypoint detection
- **sign.mt:** Translation tool (acquired by Nagish)

---

## Summary: Actionable New Sources

| # | Source | Signs/Size | Access | Video? | License | Feasibility |
|---|--------|-----------|--------|--------|---------|-------------|
| 1 | SignBSL.com | 21,000+ videos, 150+ signers | Web scraping | Yes (JS-loaded) | Copyright, contact owner | HIGH |
| 2 | Dicta-Sign BSL | ~1,000 concepts | Web browsing | Yes | EU project, contact Hamburg | MEDIUM-HIGH |
| 3 | SSC STEM Glossaries | ~2,000+ terms | Web scraping | Yes | Univ Edinburgh IP | MEDIUM |
| 4 | BSL-1K | 1,000 signs, 1,000 hours | Via BOBSL | Yes | BBC Terms of Use | LOW (licensing) |
| 5 | BANZ-FS | 35,000+ fingerspelling | Download | Yes | CC BY 4.0 | HIGH |
| 6 | SpreadTheSign | 400K+ globally | Web, needs licence | Yes | Contact ESLC | MEDIUM |
| 7 | YouTube-BSL Pipeline | Potentially 1000s | Build pipeline | Yes | Fair use / research | MEDIUM |
| 8 | YouTube-SL-25 | Unknown BSL subset | Contact authors | Yes | Research | MEDIUM |
| 9 | BSL Corpus (CAVA) | 249 signers, continuous | Open + restricted | Yes | End User License | MEDIUM |
| 10 | SignHealth videos | 100+ health topics | YouTube/Vimeo | Yes | Contact SignHealth | LOW |
| 11 | BSL Bible | Gospel of Mark | YouTube | Yes | Contact BSL Bible | LOW |
| 12 | Lumo TV | Dozens of shows | Streaming | Yes | Contact Lumo TV | LOW |

## Priority Recommendations

### Immediate (can start now):

1. **SignBSL.com extended scraping** -- Already using BSLDict (14K clips from signbsl.com). The site has grown to 21,000+ videos. Use Playwright to load pages and extract dynamically-loaded video URLs. Contact Daniel Mitchell about expanded licensing.

2. **Dicta-Sign BSL lexicon** -- ~1,000 concepts with BSL video, structured and labelled. Scrape the concept pages (URL pattern: `/cs/cs_[1-1000].html`). Contact Hamburg for training data recordings.

3. **BANZ-FS fingerspelling** -- 35K+ instances with CC BY 4.0 license. Download immediately for fingerspelling model training.

### Medium-term (requires permission/engineering):

4. **SSC BSL STEM Glossaries** -- 2,000+ domain-specific signs with video. Contact University of Edinburgh for permission. Systematic URL structure makes scraping straightforward.

5. **SpreadTheSign BSL subset** -- Large vocabulary. Contact European Sign Language Center for research licence.

6. **YouTube-BSL pipeline** -- Apply the YouTube-ASL methodology to BSL content. Identify major BSL YouTube channels, download with captions, use mouthing cue detection.

### Long-term (restrictive access):

7. **BOBSL** -- 1,400 hours, 25K annotated isolated signs. Apply via BBC R&D if academic access is possible.

8. **BSL Corpus (CAVA restricted)** -- Full conversation data with ELAN annotations. Apply for End User License.
