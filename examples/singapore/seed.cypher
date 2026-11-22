// Create the Singapore graph in LLM Graph Builder's layout, with one planted defect per metric.

CREATE (document:Document {fileName: 'singapore.txt'})
CREATE (chunk1:Chunk {text: "Lee Kuan Yew was born in Singapore. He co-founded the People's Action Party and the law firm Lee & Lee.", embedding: [0.1, 0.2]})
CREATE (chunk2:Chunk {text: 'Kwa Geok Choo married Lee Kuan Yew.', embedding: [0.3, 0.4]})
CREATE (chunk3:Chunk {text: 'Table of contents'})
CREATE (chunk1)-[:PART_OF]->(document), (chunk2)-[:PART_OF]->(document), (chunk3)-[:PART_OF]->(document)
CREATE (chunk1)-[:NEXT_CHUNK]->(chunk2)-[:NEXT_CHUNK]->(chunk3)

CREATE (lee:__Entity__:Person {id: 'Lee Kuan Yew', born: date('1923-09-16')})
CREATE (singapore:__Entity__:Location {id: 'Singapore'})
CREATE (pap:__Entity__:Organization {id: "People's Action Party"})
CREATE (unnamed:__Entity__:Organization)
CREATE (kwa:__Entity__:Person {id: 'Kwa Geok Choo', born: '1920'})
CREATE (duplicate:__Entity__:Person {id: 'lee kuan yew', born: date('1923-09-16')})
CREATE (marina_bay_sands:__Entity__:Landmark {id: 'Marina Bay Sands'})
CREATE (he:__Entity__:PERSON {id: 'He'})

CREATE (chunk1)-[:HAS_ENTITY]->(lee), (chunk1)-[:HAS_ENTITY]->(singapore), (chunk1)-[:HAS_ENTITY]->(pap), (chunk1)-[:HAS_ENTITY]->(unnamed), (chunk1)-[:HAS_ENTITY]->(he)
CREATE (chunk2)-[:HAS_ENTITY]->(lee), (chunk2)-[:HAS_ENTITY]->(kwa), (chunk2)-[:HAS_ENTITY]->(duplicate)

CREATE (lee)-[:BORN_IN]->(singapore), (lee)-[:BORN_IN]->(singapore)
CREATE (lee)-[:FOUNDED]->(pap), (lee)-[:FOUNDED]->(unnamed), (duplicate)-[:FOUNDED]->(pap)
CREATE (he)-[:Founded]->(pap)
CREATE (kwa)-[:MARRIED]->(lee), (kwa)-[:KNOWS]->(kwa)

// Plant the second document's defects, such as a surname-only duplicate and an unextracted fact.
CREATE (document2:Document {fileName: 'early-years.txt'})
CREATE (chunk4:Chunk {text: 'Lee studied law at Fitzwilliam College, Cambridge. As a boy, Lee Kuan Yew attended Raffles Institution.', embedding: [0.5, 0.6]})
CREATE (chunk5:Chunk {text: 'Raffles founded modern Singapore in 1819. Lee Kuan Yew became its first prime minister in 1959.', embedding: [0.7, 0.8]})
CREATE (chunk4)-[:PART_OF]->(document2), (chunk5)-[:PART_OF]->(document2)
CREATE (chunk4)-[:NEXT_CHUNK]->(chunk5)
CREATE (surname:__Entity__:Person {id: 'Lee'})
CREATE (fitzwilliam:__Entity__:Person {id: 'Fitzwilliam College'})
CREATE (raffles:__Entity__:Organization {id: 'Raffles'})
CREATE (chunk4)-[:HAS_ENTITY]->(lee), (chunk4)-[:HAS_ENTITY]->(surname), (chunk4)-[:HAS_ENTITY]->(fitzwilliam), (chunk4)-[:HAS_ENTITY]->(raffles)
CREATE (chunk5)-[:HAS_ENTITY]->(raffles), (chunk5)-[:HAS_ENTITY]->(singapore)
CREATE (surname)-[:STUDIED_AT]->(fitzwilliam), (lee)-[:ATTENDED]->(raffles), (raffles)-[:FOUNDED]->(singapore)
CREATE (community:__Community__ {id: '0-0', level: 0, title: "Lee Kuan Yew and the People's Action Party"})
CREATE (lee)-[:IN_COMMUNITY]->(community), (singapore)-[:IN_COMMUNITY]->(community), (pap)-[:IN_COMMUNITY]->(community), (unnamed)-[:IN_COMMUNITY]->(community)
CREATE (kwa)-[:IN_COMMUNITY]->(community), (duplicate)-[:IN_COMMUNITY]->(community), (he)-[:IN_COMMUNITY]->(community);
