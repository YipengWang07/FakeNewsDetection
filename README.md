# FakeNewsDetection

## Features

|feature|explaination|
|---|---|
| $ | symbol, currency |
| '' | closing quotation mark |
| , | punctuation mark, comma |
| -LRB- | left round bracket |
| -RRB- | right round bracket |
| . | punctuation mark, sentence closer |
| : | punctuation mark, colon or ellipsis |
| ADD | email |
| AFX | affix |
| CC | conjunction, coordinating |
| CD | cardinal number |
| DT | determiner |
| EX | existential there |
| FW | foreign word |
| HYPH | punctuation mark, hyphen |
| IN | conjunction, subordinating or preposition |
| JJ | adjective (English), other noun-modifier (Chinese) |
| JJR | adjective, comparative |
| JJS | adjective, superlative |
| LS | list item marker |
| MD | verb, modal auxiliary |
| NFP | superfluous punctuation |
| NN | noun, singular or mass |
| NNP | noun, proper singular |
| NNPS | noun, proper plural |
| NNS | noun, plural |
| PDT | predeterminer |
| POS | possessive ending |
| PRP | pronoun, personal |
| PRP$ | pronoun, possessive |
| RB | adverb |
| RBR | adverb, comparative |
| RBS | adverb, superlative |
| RP | adverb, particle |
| SYM | symbol |
| TO | infinitival "to" |
| UH | interjection |
| VB | verb, base form |
| VBD | verb, past tense |
| VBG | verb, gerund or present participle |
| VBN | verb, past participle |
| VBP | verb, non-3rd person singular present |
| VBZ | verb, 3rd person singular present |
| WDT | wh-determiner |
| WP | wh-pronoun, personal |
| WP$ | wh-pronoun, possessive |
| WRB | wh-adverb |
| XX | unknown |
| `` | opening quotation mark |
| CARDINAL | Numerals that do not fall under another type |
| DATE | Absolute or relative dates or periods |
| EVENT | Named hurricanes, battles, wars, sports events, etc. |
| FAC | Buildings, airports, highways, bridges, etc. |
| GPE | Countries, cities, states |
| LANGUAGE | Any named language |
| LAW | Named documents made into laws. |
| LOC | Non-GPE locations, mountain ranges, bodies of water |
| MONEY | Monetary values, including unit |
| NORP | Nationalities or religious or political groups |
| ORDINAL | "first", "second", etc. |
| ORG | Companies, agencies, institutions, etc. |
| PERCENT | Percentage, including "%" |
| PERSON | People, including fictional |
| PRODUCT | Objects, vehicles, foods, etc. (not services) |
| QUANTITY | Measurements, as of weight or distance |
| TIME | Times smaller than a day |
| WORK_OF_ART | Titles of books, songs, etc. |
| ROOT | None |
| acl | clausal modifier of noun (adjectival clause) |
| acomp | adjectival complement |
| advcl | adverbial clause modifier |
| advmod | adverbial modifier |
| agent | agent |
| amod | adjectival modifier |
| appos | appositional modifier |
| attr | attribute |
| aux | auxiliary |
| auxpass | auxiliary (passive) |
| case | case marking |
| cc | coordinating conjunction |
| ccomp | clausal complement |
| compound | compound |
| conj | conjunct |
| csubj | clausal subject |
| csubjpass | clausal subject (passive) |
| dative | dative |
| dep | unclassified dependent |
| det | determiner |
| dobj | direct object |
| expl | expletive |
| intj | interjection |
| mark | marker |
| meta | meta modifier |
| neg | negation modifier |
| nmod | modifier of nominal |
| npadvmod | noun phrase as adverbial modifier |
| nsubj | nominal subject |
| nsubjpass | nominal subject (passive) |
| nummod | numeric modifier |
| oprd | object predicate |
| parataxis | parataxis |
| pcomp | complement of preposition |
| pobj | object of preposition |
| poss | possession modifier |
| preconj | pre-correlative conjunction |
| predet | None |
| prep | prepositional modifier |
| prt | particle |
| punct | punctuation |
| quantmod | modifier of quantifier |
| relcl | relative clause modifier |
| xcomp | open clausal complement |


## Running Intructions

Ensure directory structure is as follows:
```
├── features.py
├── raw_data
│   ├── balancedtest.csv
│   ├── fulltrain.csv
```
Then just run `python3 features.py`
