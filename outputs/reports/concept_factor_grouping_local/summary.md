# Concept-factor sweep

Pair precision: share of the top entangled pairs that join a class factor to an attribute factor (post-hoc, hidden labels). Attribute signal: the largest I(F; a | y) / H(F) over the factors. `coact/resp` is the co-activation threshold of `cospro` and the image-response threshold of `meaning`; `min` is the least factor frequency; `composite` counts factors of more than one concept.

## metashift, laion

| method | text | coact/resp | min | merge | groups | factors | composite | cross/pairs | precision | attr. signal | attr. factors | strongest attribute factor | largest factor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meaning | 0.75 | 0.00 | 0.02 | 0.70 | 448 | 53 | 45 | 1/8 | 0.12 | 0.161 | 7 | benches | bench | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.90 | 0.00 | 0.005 | 0.00 | 1267 | 565 | 136 | 0/8 | 0.00 | 0.182 | 82 | baking | kansas, colorado, florida, texas, california |
| meaning | 0.90 | 0.00 | 0.005 | 0.90 | 1267 | 545 | 140 | 0/8 | 0.00 | 0.182 | 77 | baking | kitty, kitten, cat, tabby, chats, chat, cats, kittens |
| meaning | 0.90 | 0.30 | 0.005 | 0.00 | 1268 | 565 | 136 | 0/8 | 0.00 | 0.182 | 82 | baking | kansas, colorado, florida, texas, california |
| meaning | 0.90 | 0.30 | 0.005 | 0.90 | 1268 | 545 | 140 | 0/8 | 0.00 | 0.182 | 77 | baking | kitty, kitten, cat, tabby, chats, chat, cats, kittens |
| meaning | 0.90 | 0.50 | 0.005 | 0.00 | 1276 | 566 | 133 | 0/8 | 0.00 | 0.182 | 82 | baking | sleepy, sleeps, sleep, sleeping |
| meaning | 0.90 | 0.50 | 0.005 | 0.90 | 1276 | 546 | 137 | 0/8 | 0.00 | 0.182 | 77 | baking | kitty, kitten, cat, tabby, chats, chat, cats, kittens |
| meaning | 0.75 | 0.50 | 0.005 | 0.00 | 705 | 450 | 353 | 0/8 | 0.00 | 0.174 | 62 | fences | fence | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.75 | 0.50 | 0.005 | 0.70 | 705 | 252 | 178 | 0/8 | 0.00 | 0.174 | 29 | fences | fence | tabby, chats, kittens, chat, kitty, kitten, cats, cat |
| meaning | 0.75 | 0.50 | 0.005 | 0.90 | 705 | 450 | 353 | 0/8 | 0.00 | 0.174 | 62 | fences | fence | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.00 | 0.005 | 0.00 | 718 | 397 | 241 | 0/8 | 0.00 | 0.174 | 56 | fences | fence | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.00 | 0.005 | 0.70 | 718 | 252 | 161 | 0/8 | 0.00 | 0.174 | 28 | fences | fence | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.00 | 0.005 | 0.90 | 718 | 397 | 241 | 0/8 | 0.00 | 0.174 | 56 | fences | fence | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.30 | 0.005 | 0.00 | 785 | 455 | 299 | 0/8 | 0.00 | 0.174 | 63 | fences | fence | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.30 | 0.005 | 0.70 | 785 | 275 | 184 | 0/8 | 0.00 | 0.174 | 32 | fences | fence | canine, vet, pets, dental, pet, dog, awaiting, staring |
| meaning | 0.80 | 0.30 | 0.005 | 0.90 | 785 | 455 | 299 | 0/8 | 0.00 | 0.174 | 63 | fences | fence | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.50 | 0.005 | 0.00 | 876 | 485 | 309 | 0/8 | 0.00 | 0.174 | 71 | fences | fence | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.50 | 0.005 | 0.70 | 876 | 275 | 165 | 0/8 | 0.00 | 0.174 | 33 | fences | fence | chats, chat, kitty, kitten, cats, cat, vet, treats |
| meaning | 0.80 | 0.50 | 0.005 | 0.90 | 876 | 483 | 309 | 0/8 | 0.00 | 0.174 | 71 | fences | fence | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.85 | 0.00 | 0.005 | 0.00 | 1021 | 501 | 221 | 0/8 | 0.00 | 0.174 | 72 | fences | fence | fest, clicking, properly, geek, traveler, leak, february, january |
| meaning | 0.85 | 0.00 | 0.005 | 0.70 | 1021 | 279 | 142 | 0/8 | 0.00 | 0.174 | 31 | fences | fence | sleepy, nap, asleep, sleeps, sleep, sleeping, awaiting, waiting |
| meaning | 0.85 | 0.00 | 0.005 | 0.90 | 1021 | 493 | 220 | 0/8 | 0.00 | 0.174 | 70 | fences | fence | fest, clicking, properly, geek, traveler, leak, february, january |
| meaning | 0.85 | 0.30 | 0.005 | 0.00 | 1045 | 514 | 234 | 0/8 | 0.00 | 0.174 | 72 | fences | fence | clicking, properly, intelligent, proud, powerful, hit, got, me |
| meaning | 0.85 | 0.30 | 0.005 | 0.70 | 1045 | 277 | 139 | 0/8 | 0.00 | 0.174 | 32 | fences | fence | canine, vet, pets, dental, pet, dog, pups, pup |
| meaning | 0.85 | 0.30 | 0.005 | 0.90 | 1045 | 507 | 233 | 0/8 | 0.00 | 0.174 | 70 | fences | fence | bicycles, biking, biker, bikes, cycle, cycling, bicycle, bike |
| meaning | 0.85 | 0.50 | 0.005 | 0.00 | 1085 | 526 | 233 | 0/8 | 0.00 | 0.174 | 77 | fences | fence | bicycles, biking, biker, bikes, cycle, cycling, bicycle, bike |
| meaning | 0.85 | 0.50 | 0.005 | 0.70 | 1085 | 274 | 127 | 0/8 | 0.00 | 0.174 | 33 | fences | fence | vet, pets, pet, inspecting, inspect, sleepy, nap, asleep |
| meaning | 0.85 | 0.50 | 0.005 | 0.90 | 1085 | 519 | 232 | 0/8 | 0.00 | 0.174 | 75 | fences | fence | bicycles, biking, biker, bikes, cycle, cycling, bicycle, bike |
| meaning | 0.80 | 0.00 | 0.01 | 0.00 | 718 | 229 | 167 | 0/8 | 0.00 | 0.173 | 24 | baking | cooking | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.00 | 0.01 | 0.90 | 718 | 229 | 167 | 0/8 | 0.00 | 0.173 | 24 | baking | cooking | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.30 | 0.01 | 0.00 | 785 | 265 | 204 | 0/8 | 0.00 | 0.173 | 26 | baking | cooking | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.30 | 0.01 | 0.90 | 785 | 265 | 204 | 0/8 | 0.00 | 0.173 | 26 | baking | cooking | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.50 | 0.01 | 0.00 | 876 | 275 | 207 | 0/8 | 0.00 | 0.173 | 34 | baking | cooking | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.50 | 0.01 | 0.90 | 876 | 273 | 207 | 0/8 | 0.00 | 0.173 | 34 | baking | cooking | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.85 | 0.00 | 0.01 | 0.00 | 1021 | 271 | 152 | 0/8 | 0.00 | 0.173 | 27 | baking | cooking | fest, clicking, properly, geek, traveler, leak, february, january |
| meaning | 0.85 | 0.00 | 0.01 | 0.90 | 1021 | 265 | 151 | 0/8 | 0.00 | 0.173 | 27 | baking | cooking | fest, clicking, properly, geek, traveler, leak, february, january |
| meaning | 0.85 | 0.30 | 0.01 | 0.00 | 1045 | 280 | 160 | 0/8 | 0.00 | 0.173 | 27 | baking | cooking | clicking, properly, intelligent, proud, powerful, hit, got, me |
| meaning | 0.85 | 0.30 | 0.01 | 0.90 | 1045 | 275 | 159 | 0/8 | 0.00 | 0.173 | 27 | baking | cooking | bicycles, biking, biker, bikes, cycle, cycling, bicycle, bike |
| meaning | 0.85 | 0.50 | 0.01 | 0.00 | 1085 | 278 | 156 | 0/8 | 0.00 | 0.173 | 31 | baking | cooking | bicycles, biking, biker, bikes, cycle, cycling, bicycle, bike |
| meaning | 0.85 | 0.50 | 0.01 | 0.90 | 1085 | 273 | 155 | 0/8 | 0.00 | 0.173 | 31 | baking | cooking | bicycles, biking, biker, bikes, cycle, cycling, bicycle, bike |
| meaning | 0.75 | 0.00 | 0.005 | 0.00 | 448 | 284 | 207 | 0/8 | 0.00 | 0.165 | 39 | patches | charms | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.75 | 0.00 | 0.005 | 0.70 | 448 | 187 | 134 | 0/8 | 0.00 | 0.165 | 24 | patches | charms | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.75 | 0.00 | 0.005 | 0.90 | 448 | 284 | 207 | 0/8 | 0.00 | 0.165 | 39 | patches | charms | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.75 | 0.30 | 0.005 | 0.00 | 549 | 379 | 305 | 0/8 | 0.00 | 0.165 | 45 | patches | charms | coding, gamer, geek, blogging, cyber, editing, programming, editor |
| meaning | 0.75 | 0.30 | 0.005 | 0.70 | 549 | 228 | 177 | 0/8 | 0.00 | 0.165 | 24 | patches | charms | paws, tabby, paw, kitty, kitten, bunny, lion, rabbit |
| meaning | 0.75 | 0.30 | 0.005 | 0.90 | 549 | 379 | 305 | 0/8 | 0.00 | 0.165 | 45 | patches | charms | coding, gamer, geek, blogging, cyber, editing, programming, editor |
| meaning | 0.90 | 0.00 | 0.01 | 0.00 | 1267 | 291 | 94 | 0/8 | 0.00 | 0.162 | 32 | auto | car | kansas, colorado, florida, texas, california |
| meaning | 0.90 | 0.00 | 0.01 | 0.70 | 1267 | 131 | 67 | 0/8 | 0.00 | 0.162 | 11 | auto | car | vet, pets, pet, inspecting, sleepy, sleeps, sleep, sleeping |
| meaning | 0.90 | 0.00 | 0.01 | 0.90 | 1267 | 274 | 98 | 0/8 | 0.00 | 0.162 | 29 | auto | car | kitty, kitten, cat, tabby, chats, chat, cats, kittens |
| meaning | 0.90 | 0.00 | 0.005 | 0.70 | 1267 | 279 | 91 | 0/8 | 0.00 | 0.162 | 33 | auto | car | kitty, kitten, cat, vet, pets, pet, tabby, chats |
| meaning | 0.90 | 0.30 | 0.01 | 0.00 | 1268 | 291 | 94 | 0/8 | 0.00 | 0.162 | 32 | auto | car | kansas, colorado, florida, texas, california |
| meaning | 0.90 | 0.30 | 0.01 | 0.70 | 1268 | 131 | 67 | 0/8 | 0.00 | 0.162 | 11 | auto | car | vet, pets, pet, inspecting, sleepy, sleeps, sleep, sleeping |
| meaning | 0.90 | 0.30 | 0.01 | 0.90 | 1268 | 274 | 98 | 0/8 | 0.00 | 0.162 | 29 | auto | car | kitty, kitten, cat, tabby, chats, chat, cats, kittens |
| meaning | 0.90 | 0.30 | 0.005 | 0.70 | 1268 | 279 | 91 | 0/8 | 0.00 | 0.162 | 33 | auto | car | kitty, kitten, cat, vet, pets, pet, tabby, chats |
| meaning | 0.90 | 0.50 | 0.01 | 0.00 | 1276 | 290 | 91 | 0/8 | 0.00 | 0.162 | 32 | auto | car | sleepy, sleeps, sleep, sleeping |
| meaning | 0.90 | 0.50 | 0.01 | 0.70 | 1276 | 129 | 64 | 0/8 | 0.00 | 0.162 | 11 | auto | car | vet, pets, pet, inspecting, sleepy, sleeps, sleep, sleeping |
| meaning | 0.90 | 0.50 | 0.01 | 0.90 | 1276 | 273 | 95 | 0/8 | 0.00 | 0.162 | 29 | auto | car | kitty, kitten, cat, tabby, chats, chat, cats, kittens |
| meaning | 0.90 | 0.50 | 0.005 | 0.70 | 1276 | 280 | 89 | 0/8 | 0.00 | 0.162 | 33 | auto | car | kitty, kitten, cat, vet, pets, pet, tabby, chats |
| cospro | 0.60 | 0.10 | 0.02 | 0.00 | 249 | 107 | 19 | 0/8 | 0.00 | 0.161 | 14 | benches | bench | chats, kittens, chat, kitty, kitten, cats, cat |
| cospro | 0.60 | 0.10 | 0.02 | 0.70 | 249 | 41 | 13 | 0/8 | 0.00 | 0.161 | 6 | benches | bench | pets, pet, sleepy, asleep, sleep, sleeping, watching, cosy |
| cospro | 0.60 | 0.10 | 0.02 | 0.90 | 249 | 105 | 20 | 0/8 | 0.00 | 0.161 | 14 | benches | bench | chats, kittens, chat, kitty, kitten, cats, cat |
| cospro | 0.60 | 0.10 | 0.01 | 0.00 | 249 | 249 | 23 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | chats, kittens, chat, kitty, kitten, cats, cat |
| cospro | 0.60 | 0.10 | 0.01 | 0.70 | 249 | 116 | 21 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | pets, pet, inspecting, sleepy, asleep, sleep, sleeping, watching |
| cospro | 0.60 | 0.10 | 0.01 | 0.90 | 249 | 246 | 25 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | chats, kittens, chat, kitty, kitten, cats, cat |
| cospro | 0.60 | 0.10 | 0.005 | 0.00 | 249 | 249 | 23 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | chats, kittens, chat, kitty, kitten, cats, cat |
| cospro | 0.60 | 0.10 | 0.005 | 0.70 | 249 | 116 | 21 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | pets, pet, inspecting, sleepy, asleep, sleep, sleeping, watching |
| cospro | 0.60 | 0.10 | 0.005 | 0.90 | 249 | 246 | 25 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | chats, kittens, chat, kitty, kitten, cats, cat |
| cospro | 0.60 | 0.30 | 0.02 | 0.00 | 265 | 117 | 17 | 0/8 | 0.00 | 0.161 | 15 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.60 | 0.30 | 0.02 | 0.70 | 265 | 41 | 11 | 0/8 | 0.00 | 0.161 | 6 | benches | bench | pets, pet, watching, sleepy, bedside, bed, sleeping, vet |
| cospro | 0.60 | 0.30 | 0.02 | 0.90 | 265 | 104 | 19 | 0/8 | 0.00 | 0.161 | 14 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| cospro | 0.60 | 0.30 | 0.01 | 0.00 | 265 | 265 | 19 | 0/8 | 0.00 | 0.161 | 24 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.60 | 0.30 | 0.01 | 0.70 | 265 | 116 | 22 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | pets, pet, inspecting, watching, sleepy, bedside, bed, sleeping |
| cospro | 0.60 | 0.30 | 0.01 | 0.90 | 265 | 246 | 25 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| cospro | 0.60 | 0.30 | 0.005 | 0.00 | 265 | 265 | 19 | 0/8 | 0.00 | 0.161 | 24 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.60 | 0.30 | 0.005 | 0.70 | 265 | 116 | 22 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | pets, pet, inspecting, watching, sleepy, bedside, bed, sleeping |
| cospro | 0.60 | 0.30 | 0.005 | 0.90 | 265 | 246 | 25 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| cospro | 0.80 | 0.10 | 0.02 | 0.00 | 265 | 116 | 15 | 0/8 | 0.00 | 0.161 | 16 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.80 | 0.10 | 0.02 | 0.70 | 265 | 40 | 10 | 0/8 | 0.00 | 0.161 | 5 | benches | bench | pets, pet, watching, sleepy, sleeping, vet, helper, cosy |
| cospro | 0.80 | 0.10 | 0.02 | 0.90 | 265 | 103 | 18 | 0/8 | 0.00 | 0.161 | 14 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| cospro | 0.80 | 0.10 | 0.01 | 0.00 | 265 | 265 | 17 | 0/8 | 0.00 | 0.161 | 25 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.80 | 0.10 | 0.01 | 0.70 | 265 | 117 | 22 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | pets, pet, inspecting, watching, sleepy, sleeping, vet, helper |
| cospro | 0.80 | 0.10 | 0.01 | 0.90 | 265 | 248 | 23 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| cospro | 0.80 | 0.10 | 0.005 | 0.00 | 265 | 265 | 17 | 0/8 | 0.00 | 0.161 | 25 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.80 | 0.10 | 0.005 | 0.70 | 265 | 117 | 22 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | pets, pet, inspecting, watching, sleepy, sleeping, vet, helper |
| cospro | 0.80 | 0.10 | 0.005 | 0.90 | 265 | 248 | 23 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| cospro | 0.80 | 0.30 | 0.02 | 0.00 | 268 | 117 | 14 | 0/8 | 0.00 | 0.161 | 16 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.80 | 0.30 | 0.02 | 0.70 | 268 | 40 | 9 | 0/8 | 0.00 | 0.161 | 5 | benches | bench | pets, pet, watching, sleepy, sleeping, vet, helper, cosy |
| cospro | 0.80 | 0.30 | 0.02 | 0.90 | 268 | 103 | 17 | 0/8 | 0.00 | 0.161 | 14 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| cospro | 0.80 | 0.30 | 0.01 | 0.00 | 268 | 268 | 16 | 0/8 | 0.00 | 0.161 | 25 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.80 | 0.30 | 0.01 | 0.70 | 268 | 117 | 22 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | pets, pet, inspecting, watching, sleepy, sleeping, vet, helper |
| cospro | 0.80 | 0.30 | 0.01 | 0.90 | 268 | 248 | 23 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| cospro | 0.80 | 0.30 | 0.005 | 0.00 | 268 | 268 | 16 | 0/8 | 0.00 | 0.161 | 25 | benches | bench | pups, pup, puppies, puppy |
| cospro | 0.80 | 0.30 | 0.005 | 0.70 | 268 | 117 | 22 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | pets, pet, inspecting, watching, sleepy, sleeping, vet, helper |
| cospro | 0.80 | 0.30 | 0.005 | 0.90 | 268 | 248 | 23 | 0/8 | 0.00 | 0.161 | 23 | benches | bench | cats, cat, kittens, kitten, tabby, chats, chat, kitty |
| meaning | 0.75 | 0.00 | 0.02 | 0.00 | 448 | 93 | 77 | 0/8 | 0.00 | 0.161 | 11 | benches | bench | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.75 | 0.00 | 0.02 | 0.90 | 448 | 93 | 77 | 0/8 | 0.00 | 0.161 | 11 | benches | bench | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.75 | 0.00 | 0.01 | 0.00 | 448 | 175 | 143 | 0/8 | 0.00 | 0.161 | 21 | benches | bench | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.75 | 0.00 | 0.01 | 0.70 | 448 | 109 | 92 | 0/8 | 0.00 | 0.161 | 13 | benches | bench | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.75 | 0.00 | 0.01 | 0.90 | 448 | 175 | 143 | 0/8 | 0.00 | 0.161 | 21 | benches | bench | streams, soothing, scanning, serene, awaiting, attempting, staring, springtime |
| meaning | 0.75 | 0.30 | 0.02 | 0.00 | 549 | 140 | 128 | 0/8 | 0.00 | 0.161 | 15 | benches | bench | coding, gamer, geek, blogging, cyber, editing, programming, editor |
| meaning | 0.75 | 0.30 | 0.02 | 0.70 | 549 | 64 | 58 | 0/8 | 0.00 | 0.161 | 6 | benches | bench | paws, tabby, paw, kitty, kitten, bunny, lion, rabbit |
| meaning | 0.75 | 0.30 | 0.02 | 0.90 | 549 | 140 | 128 | 0/8 | 0.00 | 0.161 | 15 | benches | bench | coding, gamer, geek, blogging, cyber, editing, programming, editor |
| meaning | 0.75 | 0.30 | 0.01 | 0.00 | 549 | 248 | 220 | 0/8 | 0.00 | 0.161 | 26 | benches | bench | coding, gamer, geek, blogging, cyber, editing, programming, editor |
| meaning | 0.75 | 0.30 | 0.01 | 0.70 | 549 | 139 | 124 | 0/8 | 0.00 | 0.161 | 11 | benches | bench | paws, tabby, paw, kitty, kitten, bunny, lion, rabbit |
| meaning | 0.75 | 0.30 | 0.01 | 0.90 | 549 | 248 | 220 | 0/8 | 0.00 | 0.161 | 26 | benches | bench | coding, gamer, geek, blogging, cyber, editing, programming, editor |
| meaning | 0.75 | 0.50 | 0.02 | 0.00 | 705 | 130 | 118 | 0/8 | 0.00 | 0.161 | 16 | benches | bench | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.75 | 0.50 | 0.02 | 0.70 | 705 | 53 | 47 | 0/8 | 0.00 | 0.161 | 7 | benches | bench | tabby, chats, kittens, chat, kitty, kitten, cats, cat |
| meaning | 0.75 | 0.50 | 0.02 | 0.90 | 705 | 130 | 118 | 0/8 | 0.00 | 0.161 | 16 | benches | bench | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.75 | 0.50 | 0.01 | 0.00 | 705 | 269 | 235 | 0/8 | 0.00 | 0.161 | 34 | benches | bench | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.75 | 0.50 | 0.01 | 0.70 | 705 | 127 | 105 | 0/8 | 0.00 | 0.161 | 11 | benches | bench | tabby, chats, kittens, chat, kitty, kitten, cats, cat |
| meaning | 0.75 | 0.50 | 0.01 | 0.90 | 705 | 269 | 235 | 0/8 | 0.00 | 0.161 | 34 | benches | bench | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.00 | 0.02 | 0.00 | 718 | 114 | 86 | 0/8 | 0.00 | 0.161 | 15 | benches | bench | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.00 | 0.02 | 0.70 | 718 | 63 | 55 | 0/8 | 0.00 | 0.161 | 5 | benches | bench | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.00 | 0.02 | 0.90 | 718 | 114 | 86 | 0/8 | 0.00 | 0.161 | 15 | benches | bench | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.00 | 0.01 | 0.70 | 718 | 142 | 114 | 0/8 | 0.00 | 0.161 | 8 | benches | bench | nuit, canine, sleepy, cruising, marie, fest, comparing, storing |
| meaning | 0.80 | 0.30 | 0.02 | 0.00 | 785 | 143 | 115 | 0/8 | 0.00 | 0.161 | 17 | benches | bench | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.30 | 0.02 | 0.70 | 785 | 67 | 59 | 0/8 | 0.00 | 0.161 | 7 | benches | bench | canine, vet, pets, dental, pet, dog, awaiting, staring |
| meaning | 0.80 | 0.30 | 0.02 | 0.90 | 785 | 143 | 115 | 0/8 | 0.00 | 0.161 | 17 | benches | bench | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.30 | 0.01 | 0.70 | 785 | 146 | 121 | 0/8 | 0.00 | 0.161 | 10 | benches | bench | canine, vet, pets, dental, pet, dog, awaiting, staring |
| meaning | 0.80 | 0.50 | 0.02 | 0.00 | 876 | 130 | 98 | 0/8 | 0.00 | 0.161 | 20 | benches | bench | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.50 | 0.02 | 0.70 | 876 | 57 | 48 | 0/8 | 0.00 | 0.161 | 8 | benches | bench | chats, chat, kitty, kitten, cats, cat, vet, treats |
| meaning | 0.80 | 0.50 | 0.02 | 0.90 | 876 | 128 | 98 | 0/8 | 0.00 | 0.161 | 20 | benches | bench | cyclists, cyclist, bicycles, motorbike, biking, motorcycles, biker, bikes |
| meaning | 0.80 | 0.50 | 0.01 | 0.70 | 876 | 135 | 105 | 0/8 | 0.00 | 0.161 | 14 | benches | bench | chats, chat, kitty, kitten, cats, cat, vet, treats |
| meaning | 0.85 | 0.00 | 0.02 | 0.00 | 1021 | 133 | 80 | 0/8 | 0.00 | 0.161 | 16 | benches | bench | fest, clicking, properly, geek, traveler, leak, february, january |
| meaning | 0.85 | 0.00 | 0.02 | 0.70 | 1021 | 69 | 53 | 0/8 | 0.00 | 0.161 | 7 | benches | bench | sleepy, nap, asleep, sleeps, sleep, sleeping, awaiting, waiting |
| meaning | 0.85 | 0.00 | 0.02 | 0.90 | 1021 | 128 | 80 | 0/8 | 0.00 | 0.161 | 16 | benches | bench | fest, clicking, properly, geek, traveler, leak, february, january |
| meaning | 0.85 | 0.00 | 0.01 | 0.70 | 1021 | 151 | 103 | 0/8 | 0.00 | 0.161 | 9 | benches | bench | sleepy, nap, asleep, sleeps, sleep, sleeping, awaiting, waiting |
| meaning | 0.85 | 0.30 | 0.02 | 0.00 | 1045 | 134 | 80 | 0/8 | 0.00 | 0.161 | 16 | benches | bench | clicking, properly, intelligent, proud, powerful, hit, got, me |
| meaning | 0.85 | 0.30 | 0.02 | 0.70 | 1045 | 62 | 46 | 0/8 | 0.00 | 0.161 | 8 | benches | bench | canine, vet, pets, dental, pet, dog, pups, pup |
| meaning | 0.85 | 0.30 | 0.02 | 0.90 | 1045 | 130 | 80 | 0/8 | 0.00 | 0.161 | 16 | benches | bench | clicking, properly, intelligent, proud, powerful, hit, got, me |
| meaning | 0.85 | 0.30 | 0.01 | 0.70 | 1045 | 146 | 98 | 0/8 | 0.00 | 0.161 | 11 | benches | bench | canine, vet, pets, dental, pet, dog, pups, pup |
| meaning | 0.85 | 0.50 | 0.02 | 0.00 | 1085 | 130 | 76 | 0/8 | 0.00 | 0.161 | 18 | benches | bench | bicycles, biking, biker, bikes, cycle, cycling, bicycle, bike |
| meaning | 0.85 | 0.50 | 0.02 | 0.70 | 1085 | 56 | 41 | 0/8 | 0.00 | 0.161 | 7 | benches | bench | vet, pets, pet, sleepy, nap, asleep, sleeps, sleep |
| meaning | 0.85 | 0.50 | 0.02 | 0.90 | 1085 | 126 | 76 | 0/8 | 0.00 | 0.161 | 18 | benches | bench | kitty, kitten, cat, chats, chat, cats, tabby, kittens |
| meaning | 0.85 | 0.50 | 0.01 | 0.70 | 1085 | 133 | 86 | 0/8 | 0.00 | 0.161 | 11 | benches | bench | vet, pets, pet, inspecting, inspect, sleepy, nap, asleep |
| meaning | 0.90 | 0.00 | 0.02 | 0.00 | 1267 | 124 | 41 | 0/8 | 0.00 | 0.161 | 17 | benches | bench | sleepy, sleeps, sleep, sleeping |
| meaning | 0.90 | 0.00 | 0.02 | 0.70 | 1267 | 47 | 25 | 0/8 | 0.00 | 0.161 | 6 | benches | bench | vet, pets, pet, sleepy, sleeps, sleep, sleeping, watching |
| meaning | 0.90 | 0.00 | 0.02 | 0.90 | 1267 | 113 | 43 | 0/8 | 0.00 | 0.161 | 15 | benches | bench | kitty, kitten, cat, tabby, chats, chat, cats, kittens |
| meaning | 0.90 | 0.30 | 0.02 | 0.00 | 1268 | 124 | 41 | 0/8 | 0.00 | 0.161 | 17 | benches | bench | sleepy, sleeps, sleep, sleeping |
| meaning | 0.90 | 0.30 | 0.02 | 0.70 | 1268 | 47 | 25 | 0/8 | 0.00 | 0.161 | 6 | benches | bench | vet, pets, pet, sleepy, sleeps, sleep, sleeping, watching |
| meaning | 0.90 | 0.30 | 0.02 | 0.90 | 1268 | 113 | 43 | 0/8 | 0.00 | 0.161 | 15 | benches | bench | kitty, kitten, cat, tabby, chats, chat, cats, kittens |
| meaning | 0.90 | 0.50 | 0.02 | 0.00 | 1276 | 124 | 40 | 0/8 | 0.00 | 0.161 | 17 | benches | bench | sleepy, sleeps, sleep, sleeping |
| meaning | 0.90 | 0.50 | 0.02 | 0.70 | 1276 | 46 | 24 | 0/8 | 0.00 | 0.161 | 6 | benches | bench | vet, pets, pet, sleepy, sleeps, sleep, sleeping, watching |
| meaning | 0.90 | 0.50 | 0.02 | 0.90 | 1276 | 113 | 42 | 0/8 | 0.00 | 0.161 | 15 | benches | bench | kitty, kitten, cat, tabby, chats, chat, cats, kittens |

## metashift, openimages_v7

| method | text | coact/resp | min | merge | groups | factors | composite | cross/pairs | precision | attr. signal | attr. factors | strongest attribute factor | largest factor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meaning | 0.80 | 0.00 | 0.005 | 0.00 | 2206 | 998 | 306 | 0/8 | 0.00 | 0.205 | 100 | Sunlounger | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.00 | 0.005 | 0.70 | 2206 | 355 | 97 | 0/8 | 0.00 | 0.205 | 41 | Sunlounger | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Dog crate, Cat toy |
| meaning | 0.80 | 0.00 | 0.005 | 0.90 | 2206 | 947 | 291 | 0/8 | 0.00 | 0.205 | 99 | Sunlounger | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Rex cat, American shorthair, British shorthair, European shorthair |
| meaning | 0.80 | 0.30 | 0.005 | 0.00 | 2209 | 999 | 307 | 0/8 | 0.00 | 0.205 | 101 | Sunlounger | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.30 | 0.005 | 0.70 | 2209 | 355 | 96 | 0/8 | 0.00 | 0.205 | 41 | Sunlounger | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Cat bed |
| meaning | 0.80 | 0.30 | 0.005 | 0.90 | 2209 | 945 | 289 | 0/8 | 0.00 | 0.205 | 100 | Sunlounger | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Rex cat |
| meaning | 0.80 | 0.50 | 0.005 | 0.00 | 2219 | 1004 | 306 | 0/8 | 0.00 | 0.205 | 102 | Sunlounger | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.50 | 0.005 | 0.70 | 2219 | 351 | 92 | 0/8 | 0.00 | 0.205 | 40 | Sunlounger | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Cat bed |
| meaning | 0.80 | 0.50 | 0.005 | 0.90 | 2219 | 949 | 288 | 0/8 | 0.00 | 0.205 | 101 | Sunlounger | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Rex cat |
| meaning | 0.85 | 0.00 | 0.005 | 0.00 | 2472 | 1055 | 178 | 0/8 | 0.00 | 0.205 | 102 | Sunlounger | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.00 | 0.005 | 0.70 | 2472 | 336 | 53 | 0/8 | 0.00 | 0.205 | 39 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Cat toy, Dog toy, Rex cat |
| meaning | 0.85 | 0.00 | 0.005 | 0.90 | 2472 | 976 | 179 | 0/8 | 0.00 | 0.205 | 102 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Domestic long-haired cat, Norwegian forest cat |
| meaning | 0.85 | 0.30 | 0.005 | 0.00 | 2471 | 1053 | 180 | 0/8 | 0.00 | 0.205 | 102 | Sunlounger | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.30 | 0.005 | 0.70 | 2471 | 335 | 53 | 0/8 | 0.00 | 0.205 | 39 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.85 | 0.30 | 0.005 | 0.90 | 2471 | 973 | 179 | 0/8 | 0.00 | 0.205 | 102 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Cat, Kitten, Domestic long-haired cat |
| meaning | 0.85 | 0.50 | 0.005 | 0.00 | 2472 | 1053 | 179 | 0/8 | 0.00 | 0.205 | 102 | Sunlounger | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.50 | 0.005 | 0.70 | 2472 | 337 | 53 | 0/8 | 0.00 | 0.205 | 39 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.85 | 0.50 | 0.005 | 0.90 | 2472 | 973 | 178 | 0/8 | 0.00 | 0.205 | 102 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Cat, Kitten, Domestic long-haired cat |
| meaning | 0.90 | 0.00 | 0.005 | 0.00 | 2614 | 1085 | 85 | 0/8 | 0.00 | 0.205 | 100 | Sunlounger | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.00 | 0.005 | 0.90 | 2614 | 963 | 110 | 0/8 | 0.00 | 0.205 | 99 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.90 | 0.30 | 0.005 | 0.00 | 2614 | 1085 | 85 | 0/8 | 0.00 | 0.205 | 100 | Sunlounger | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.30 | 0.005 | 0.90 | 2614 | 963 | 110 | 0/8 | 0.00 | 0.205 | 99 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.90 | 0.50 | 0.005 | 0.00 | 2614 | 1085 | 85 | 0/8 | 0.00 | 0.205 | 100 | Sunlounger | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.50 | 0.005 | 0.90 | 2614 | 963 | 110 | 0/8 | 0.00 | 0.205 | 99 | Sunlounger | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.75 | 0.00 | 0.005 | 0.00 | 1855 | 891 | 407 | 0/8 | 0.00 | 0.202 | 88 | Animal figure | Autumn, Beach, Blogger, Boxer, Cat, Creek, Desk, Dog |
| meaning | 0.75 | 0.00 | 0.005 | 0.90 | 1855 | 860 | 391 | 0/8 | 0.00 | 0.202 | 88 | Animal figure | Autumn, Beach, Blogger, Boxer, Cat, Creek, Desk, Dog |
| meaning | 0.75 | 0.30 | 0.005 | 0.00 | 1864 | 894 | 407 | 0/8 | 0.00 | 0.202 | 93 | Animal figure | American cocker spaniel, Blue picardy spaniel, Boykin spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, English setter, English springer spaniel |
| meaning | 0.75 | 0.30 | 0.005 | 0.90 | 1864 | 861 | 389 | 0/8 | 0.00 | 0.202 | 93 | Animal figure | Cat bed, Cat food, Cat furniture, Cat litter, Cat supply, Cat toy, Cat tree, Dog bed |
| meaning | 0.75 | 0.50 | 0.005 | 0.00 | 1902 | 912 | 413 | 0/8 | 0.00 | 0.202 | 99 | Animal figure | American cocker spaniel, Blue picardy spaniel, Boykin spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, English setter, English springer spaniel |
| meaning | 0.75 | 0.50 | 0.005 | 0.90 | 1902 | 878 | 395 | 0/8 | 0.00 | 0.202 | 99 | Animal figure | Domestic long-haired cat, Domestic short-haired cat, Maine coon, Norwegian forest cat, Tabby cat, Cat food, Cat furniture, Cat litter |
| cospro | 0.60 | 0.10 | 0.01 | 0.00 | 387 | 387 | 40 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | American bobtail, British longhair, British semi-longhair, Domestic long-haired cat, Kurilian bobtail, Maine coon, Norwegian forest cat |
| cospro | 0.60 | 0.10 | 0.01 | 0.70 | 387 | 107 | 10 | 0/8 | 0.00 | 0.196 | 11 | Dog bed | American shorthair, American wirehair, British shorthair, Domestic short-haired cat, European shorthair, Scottish fold, Tabby cat, Cat bed |
| cospro | 0.60 | 0.10 | 0.01 | 0.90 | 387 | 358 | 46 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | American shorthair, American wirehair, British shorthair, Domestic short-haired cat, European shorthair, Scottish fold, Tabby cat, Cat bed |
| cospro | 0.60 | 0.10 | 0.005 | 0.00 | 387 | 387 | 40 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | American bobtail, British longhair, British semi-longhair, Domestic long-haired cat, Kurilian bobtail, Maine coon, Norwegian forest cat |
| cospro | 0.60 | 0.10 | 0.005 | 0.70 | 387 | 107 | 10 | 0/8 | 0.00 | 0.196 | 11 | Dog bed | American shorthair, American wirehair, British shorthair, Domestic short-haired cat, European shorthair, Scottish fold, Tabby cat, Cat bed |
| cospro | 0.60 | 0.10 | 0.005 | 0.90 | 387 | 358 | 46 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | American shorthair, American wirehair, British shorthair, Domestic short-haired cat, European shorthair, Scottish fold, Tabby cat, Cat bed |
| cospro | 0.60 | 0.30 | 0.01 | 0.00 | 442 | 442 | 13 | 0/8 | 0.00 | 0.196 | 37 | Dog bed | Angus cattle, Gyr cattle |
| cospro | 0.60 | 0.30 | 0.01 | 0.70 | 442 | 107 | 10 | 0/8 | 0.00 | 0.196 | 11 | Dog bed | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Cat-ketch, Tortoiseshell cat |
| cospro | 0.60 | 0.30 | 0.01 | 0.90 | 442 | 377 | 35 | 0/8 | 0.00 | 0.196 | 36 | Dog bed | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Kitten, Malayan cat |
| cospro | 0.60 | 0.30 | 0.005 | 0.00 | 442 | 442 | 13 | 0/8 | 0.00 | 0.196 | 37 | Dog bed | Angus cattle, Gyr cattle |
| cospro | 0.60 | 0.30 | 0.005 | 0.70 | 442 | 107 | 10 | 0/8 | 0.00 | 0.196 | 11 | Dog bed | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Cat-ketch, Tortoiseshell cat |
| cospro | 0.60 | 0.30 | 0.005 | 0.90 | 442 | 377 | 35 | 0/8 | 0.00 | 0.196 | 36 | Dog bed | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Kitten, Malayan cat |
| cospro | 0.80 | 0.10 | 0.01 | 0.00 | 450 | 450 | 5 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | Australian collie, Border collie |
| cospro | 0.80 | 0.10 | 0.01 | 0.70 | 450 | 105 | 9 | 0/8 | 0.00 | 0.196 | 11 | Dog bed | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Cat-ketch, Tortoiseshell cat |
| cospro | 0.80 | 0.10 | 0.01 | 0.90 | 450 | 383 | 29 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Kitten, Malayan cat |
| cospro | 0.80 | 0.10 | 0.005 | 0.00 | 450 | 450 | 5 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | Australian collie, Border collie |
| cospro | 0.80 | 0.10 | 0.005 | 0.70 | 450 | 105 | 9 | 0/8 | 0.00 | 0.196 | 11 | Dog bed | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Cat-ketch, Tortoiseshell cat |
| cospro | 0.80 | 0.10 | 0.005 | 0.90 | 450 | 383 | 29 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Kitten, Malayan cat |
| cospro | 0.80 | 0.30 | 0.01 | 0.00 | 455 | 455 | 0 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | Abyssinian |
| cospro | 0.80 | 0.30 | 0.01 | 0.70 | 455 | 105 | 9 | 0/8 | 0.00 | 0.196 | 11 | Dog bed | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Cat-ketch, Tortoiseshell cat |
| cospro | 0.80 | 0.30 | 0.01 | 0.90 | 455 | 383 | 29 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Kitten, Malayan cat |
| cospro | 0.80 | 0.30 | 0.005 | 0.00 | 455 | 455 | 0 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | Abyssinian |
| cospro | 0.80 | 0.30 | 0.005 | 0.70 | 455 | 105 | 9 | 0/8 | 0.00 | 0.196 | 11 | Dog bed | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Cat-ketch, Tortoiseshell cat |
| cospro | 0.80 | 0.30 | 0.005 | 0.90 | 455 | 383 | 29 | 0/8 | 0.00 | 0.196 | 35 | Dog bed | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Kitten, Malayan cat |
| cospro | 0.60 | 0.30 | 0.02 | 0.00 | 442 | 182 | 9 | 0/8 | 0.00 | 0.193 | 17 | Bench | Animal training, Obedience training |
| cospro | 0.60 | 0.30 | 0.02 | 0.90 | 442 | 150 | 13 | 0/8 | 0.00 | 0.193 | 16 | Bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Kitten, Malayan cat |
| cospro | 0.80 | 0.10 | 0.02 | 0.00 | 450 | 184 | 5 | 0/8 | 0.00 | 0.193 | 16 | Bench | Australian collie, Border collie |
| cospro | 0.80 | 0.10 | 0.02 | 0.90 | 450 | 146 | 11 | 0/8 | 0.00 | 0.193 | 15 | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Kitten, Malayan cat |
| cospro | 0.80 | 0.30 | 0.02 | 0.00 | 455 | 182 | 0 | 0/8 | 0.00 | 0.193 | 16 | Bench | Abyssinian |
| cospro | 0.80 | 0.30 | 0.02 | 0.90 | 455 | 145 | 7 | 0/8 | 0.00 | 0.193 | 15 | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Kitten, Malayan cat |
| meaning | 0.90 | 0.00 | 0.02 | 0.00 | 2614 | 184 | 12 | 0/8 | 0.00 | 0.193 | 15 | Bench | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.00 | 0.02 | 0.90 | 2614 | 148 | 16 | 0/8 | 0.00 | 0.193 | 15 | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.90 | 0.00 | 0.01 | 0.00 | 2614 | 459 | 44 | 0/8 | 0.00 | 0.193 | 35 | Bench | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.00 | 0.01 | 0.90 | 2614 | 391 | 58 | 0/8 | 0.00 | 0.193 | 36 | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.90 | 0.30 | 0.02 | 0.00 | 2614 | 184 | 12 | 0/8 | 0.00 | 0.193 | 15 | Bench | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.30 | 0.02 | 0.90 | 2614 | 148 | 16 | 0/8 | 0.00 | 0.193 | 15 | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.90 | 0.30 | 0.01 | 0.00 | 2614 | 459 | 44 | 0/8 | 0.00 | 0.193 | 35 | Bench | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.30 | 0.01 | 0.90 | 2614 | 391 | 58 | 0/8 | 0.00 | 0.193 | 36 | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.90 | 0.50 | 0.02 | 0.00 | 2614 | 184 | 12 | 0/8 | 0.00 | 0.193 | 15 | Bench | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.50 | 0.02 | 0.90 | 2614 | 148 | 16 | 0/8 | 0.00 | 0.193 | 15 | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.90 | 0.50 | 0.01 | 0.00 | 2614 | 459 | 44 | 0/8 | 0.00 | 0.193 | 35 | Bench | Australian collie, Border collie, Collie, Scotch collie |
| meaning | 0.90 | 0.50 | 0.01 | 0.90 | 2614 | 391 | 58 | 0/8 | 0.00 | 0.193 | 36 | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Malayan cat, Arabian mau |
| meaning | 0.75 | 0.00 | 0.005 | 0.70 | 1855 | 366 | 153 | 0/8 | 0.00 | 0.179 | 40 | Car alarm | Domestic long-haired cat, Domestic short-haired cat, Maine coon, Norwegian forest cat, Tabby cat, Autumn, Beach, Blogger |
| meaning | 0.75 | 0.30 | 0.005 | 0.70 | 1864 | 362 | 148 | 0/8 | 0.00 | 0.179 | 42 | Car alarm | Cat bed, Cat food, Cat furniture, Cat litter, Cat supply, Cat toy, Cat tree, Dog bed |
| meaning | 0.75 | 0.50 | 0.005 | 0.70 | 1902 | 363 | 145 | 0/8 | 0.00 | 0.179 | 42 | Car alarm | Domestic long-haired cat, Domestic short-haired cat, Maine coon, Norwegian forest cat, Tabby cat, Cat food, Cat furniture, Cat litter |
| meaning | 0.90 | 0.00 | 0.005 | 0.70 | 2614 | 331 | 36 | 0/8 | 0.00 | 0.179 | 38 | Car alarm | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.90 | 0.30 | 0.005 | 0.70 | 2614 | 331 | 36 | 0/8 | 0.00 | 0.179 | 38 | Car alarm | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.90 | 0.50 | 0.005 | 0.70 | 2614 | 331 | 36 | 0/8 | 0.00 | 0.179 | 38 | Car alarm | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| cospro | 0.60 | 0.10 | 0.02 | 0.00 | 387 | 168 | 37 | 0/8 | 0.00 | 0.173 | 15 | Bench | Outdoor bench | American bobtail, British longhair, British semi-longhair, Domestic long-haired cat, Kurilian bobtail, Maine coon, Norwegian forest cat |
| cospro | 0.60 | 0.10 | 0.02 | 0.70 | 387 | 37 | 10 | 0/8 | 0.00 | 0.173 | 4 | Bench | Outdoor bench | American shorthair, American wirehair, British shorthair, Domestic short-haired cat, European shorthair, Scottish fold, Tabby cat, Cat bed |
| cospro | 0.60 | 0.10 | 0.02 | 0.90 | 387 | 153 | 37 | 0/8 | 0.00 | 0.173 | 15 | Bench | Outdoor bench | American shorthair, American wirehair, British shorthair, Domestic short-haired cat, European shorthair, Scottish fold, Tabby cat, Cat bed |
| cospro | 0.60 | 0.30 | 0.02 | 0.70 | 442 | 33 | 6 | 0/8 | 0.00 | 0.173 | 4 | Outdoor bench | Bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Cat-ketch, Tortoiseshell cat |
| cospro | 0.80 | 0.10 | 0.02 | 0.70 | 450 | 34 | 6 | 0/8 | 0.00 | 0.173 | 4 | Outdoor bench | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Cat-ketch, Tortoiseshell cat |
| cospro | 0.80 | 0.30 | 0.02 | 0.70 | 455 | 33 | 5 | 0/8 | 0.00 | 0.173 | 4 | Outdoor bench | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Rex cat, Tabby cat, Cat-ketch, Tortoiseshell cat |
| meaning | 0.90 | 0.00 | 0.02 | 0.70 | 2614 | 34 | 7 | 0/8 | 0.00 | 0.173 | 4 | Outdoor bench | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.90 | 0.00 | 0.01 | 0.70 | 2614 | 105 | 14 | 0/8 | 0.00 | 0.173 | 11 | Outdoor bench | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.90 | 0.30 | 0.02 | 0.70 | 2614 | 34 | 7 | 0/8 | 0.00 | 0.173 | 4 | Outdoor bench | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.90 | 0.30 | 0.01 | 0.70 | 2614 | 105 | 14 | 0/8 | 0.00 | 0.173 | 11 | Outdoor bench | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.90 | 0.50 | 0.02 | 0.70 | 2614 | 34 | 7 | 0/8 | 0.00 | 0.173 | 4 | Outdoor bench | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.90 | 0.50 | 0.01 | 0.70 | 2614 | 105 | 14 | 0/8 | 0.00 | 0.173 | 11 | Outdoor bench | Bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.75 | 0.50 | 0.02 | 0.00 | 1902 | 209 | 123 | 0/8 | 0.00 | 0.172 | 17 | Bench | Bench (Geography) | Outdoor bench (+2) | American cocker spaniel, Blue picardy spaniel, Boykin spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, English setter, English springer spaniel |
| meaning | 0.75 | 0.50 | 0.02 | 0.70 | 1902 | 42 | 23 | 0/8 | 0.00 | 0.172 | 5 | Bench | Bench (Geography) | Outdoor bench (+2) | Ancient dog breeds, Dog Breed Group, Dog crossbreeds, Giant dog breed, Small greek domestic dog, Hare coursing, Lure coursing, Dog hiking |
| meaning | 0.75 | 0.50 | 0.02 | 0.90 | 1902 | 189 | 115 | 0/8 | 0.00 | 0.172 | 17 | Bench | Bench (Geography) | Outdoor bench (+2) | Domestic long-haired cat, Domestic short-haired cat, Maine coon, Norwegian forest cat, Tabby cat, Cat food, Cat furniture, Cat litter |
| meaning | 0.75 | 0.50 | 0.01 | 0.00 | 1902 | 468 | 270 | 0/8 | 0.00 | 0.172 | 45 | Bench | Bench (Geography) | Outdoor bench (+2) | American cocker spaniel, Blue picardy spaniel, Boykin spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, English setter, English springer spaniel |
| meaning | 0.75 | 0.50 | 0.01 | 0.70 | 1902 | 129 | 73 | 0/8 | 0.00 | 0.172 | 13 | Bench | Bench (Geography) | Outdoor bench (+2) | Domestic long-haired cat, Domestic short-haired cat, Maine coon, Norwegian forest cat, Tabby cat, Cat food, Cat furniture, Cat litter |
| meaning | 0.75 | 0.50 | 0.01 | 0.90 | 1902 | 443 | 258 | 0/8 | 0.00 | 0.172 | 45 | Bench | Bench (Geography) | Outdoor bench (+2) | Domestic long-haired cat, Domestic short-haired cat, Maine coon, Norwegian forest cat, Tabby cat, Cat food, Cat furniture, Cat litter |
| meaning | 0.75 | 0.00 | 0.02 | 0.00 | 1855 | 210 | 124 | 0/8 | 0.00 | 0.169 | 15 | Bench | Bench (Geography) | Outdoor bench | Autumn, Beach, Blogger, Boxer, Cat, Creek, Desk, Dog |
| meaning | 0.75 | 0.00 | 0.02 | 0.70 | 1855 | 47 | 28 | 0/8 | 0.00 | 0.169 | 5 | Bench | Bench (Geography) | Outdoor bench | Ancient dog breeds, Dog Breed Group, Dog crossbreeds, Giant dog breed, Small greek domestic dog, Hare coursing, Lure coursing, Dog hiking |
| meaning | 0.75 | 0.00 | 0.02 | 0.90 | 1855 | 194 | 118 | 0/8 | 0.00 | 0.169 | 15 | Bench | Bench (Geography) | Outdoor bench | Autumn, Beach, Blogger, Boxer, Cat, Creek, Desk, Dog |
| meaning | 0.75 | 0.00 | 0.01 | 0.00 | 1855 | 468 | 273 | 0/8 | 0.00 | 0.169 | 40 | Bench | Bench (Geography) | Outdoor bench | Autumn, Beach, Blogger, Boxer, Cat, Creek, Desk, Dog |
| meaning | 0.75 | 0.00 | 0.01 | 0.70 | 1855 | 136 | 82 | 0/8 | 0.00 | 0.169 | 13 | Bench | Bench (Geography) | Outdoor bench | Domestic long-haired cat, Domestic short-haired cat, Maine coon, Norwegian forest cat, Tabby cat, Autumn, Beach, Blogger |
| meaning | 0.75 | 0.00 | 0.01 | 0.90 | 1855 | 446 | 263 | 0/8 | 0.00 | 0.169 | 40 | Bench | Bench (Geography) | Outdoor bench | Autumn, Beach, Blogger, Boxer, Cat, Creek, Desk, Dog |
| meaning | 0.75 | 0.30 | 0.02 | 0.00 | 1864 | 212 | 126 | 0/8 | 0.00 | 0.169 | 15 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Blue picardy spaniel, Boykin spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, English setter, English springer spaniel |
| meaning | 0.75 | 0.30 | 0.02 | 0.70 | 1864 | 46 | 27 | 0/8 | 0.00 | 0.169 | 4 | Bench | Bench (Geography) | Outdoor bench | Ancient dog breeds, Dog Breed Group, Dog crossbreeds, Giant dog breed, Small greek domestic dog, Hare coursing, Lure coursing, Dog hiking |
| meaning | 0.75 | 0.30 | 0.02 | 0.90 | 1864 | 194 | 118 | 0/8 | 0.00 | 0.169 | 15 | Bench | Bench (Geography) | Outdoor bench | Cat bed, Cat food, Cat furniture, Cat litter, Cat supply, Cat toy, Cat tree, Dog bed |
| meaning | 0.75 | 0.30 | 0.01 | 0.00 | 1864 | 472 | 275 | 0/8 | 0.00 | 0.169 | 42 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Blue picardy spaniel, Boykin spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, English setter, English springer spaniel |
| meaning | 0.75 | 0.30 | 0.01 | 0.70 | 1864 | 131 | 76 | 0/8 | 0.00 | 0.169 | 13 | Bench | Bench (Geography) | Outdoor bench | Cat bed, Cat food, Cat furniture, Cat litter, Cat supply, Cat toy, Cat tree, Dog bed |
| meaning | 0.75 | 0.30 | 0.01 | 0.90 | 1864 | 448 | 263 | 0/8 | 0.00 | 0.169 | 42 | Bench | Bench (Geography) | Outdoor bench | Cat bed, Cat food, Cat furniture, Cat litter, Cat supply, Cat toy, Cat tree, Dog bed |
| meaning | 0.80 | 0.00 | 0.02 | 0.00 | 2206 | 203 | 85 | 0/8 | 0.00 | 0.169 | 13 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.00 | 0.02 | 0.70 | 2206 | 41 | 15 | 0/8 | 0.00 | 0.169 | 5 | Bench | Bench (Geography) | Outdoor bench | Hare coursing, Lure coursing, Dog hiking, Dog walking, Street dog, Ancient dog breeds, Dog Breed Group, Dog crossbreeds |
| meaning | 0.80 | 0.00 | 0.02 | 0.90 | 2206 | 185 | 81 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Rex cat, American shorthair, British shorthair, European shorthair |
| meaning | 0.80 | 0.00 | 0.01 | 0.00 | 2206 | 475 | 198 | 0/8 | 0.00 | 0.169 | 35 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.00 | 0.01 | 0.70 | 2206 | 123 | 51 | 0/8 | 0.00 | 0.169 | 13 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Dog crate, Cat toy |
| meaning | 0.80 | 0.00 | 0.01 | 0.90 | 2206 | 440 | 186 | 0/8 | 0.00 | 0.169 | 36 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Rex cat, American shorthair, British shorthair, European shorthair |
| meaning | 0.80 | 0.30 | 0.02 | 0.00 | 2209 | 202 | 84 | 0/8 | 0.00 | 0.169 | 13 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.30 | 0.02 | 0.70 | 2209 | 39 | 12 | 0/8 | 0.00 | 0.169 | 5 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Cat bed |
| meaning | 0.80 | 0.30 | 0.02 | 0.90 | 2209 | 181 | 77 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Rex cat |
| meaning | 0.80 | 0.30 | 0.01 | 0.00 | 2209 | 474 | 197 | 0/8 | 0.00 | 0.169 | 35 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.30 | 0.01 | 0.70 | 2209 | 122 | 49 | 0/8 | 0.00 | 0.169 | 13 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Cat bed |
| meaning | 0.80 | 0.30 | 0.01 | 0.90 | 2209 | 436 | 182 | 0/8 | 0.00 | 0.169 | 36 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Rex cat |
| meaning | 0.80 | 0.50 | 0.02 | 0.00 | 2219 | 201 | 83 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.50 | 0.02 | 0.70 | 2219 | 38 | 11 | 0/8 | 0.00 | 0.169 | 5 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Cat bed |
| meaning | 0.80 | 0.50 | 0.02 | 0.90 | 2219 | 179 | 76 | 0/8 | 0.00 | 0.169 | 15 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Rex cat |
| meaning | 0.80 | 0.50 | 0.01 | 0.00 | 2219 | 474 | 195 | 0/8 | 0.00 | 0.169 | 36 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Blue picardy spaniel, Cavalier king charles spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, King charles spaniel |
| meaning | 0.80 | 0.50 | 0.01 | 0.70 | 2219 | 118 | 45 | 0/8 | 0.00 | 0.169 | 13 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Cat bed |
| meaning | 0.80 | 0.50 | 0.01 | 0.90 | 2219 | 435 | 180 | 0/8 | 0.00 | 0.169 | 37 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Tabby cat, Polydactyl cat, Napoleon cat, Cat furniture, Cat toy, Cat tree, Rex cat |
| meaning | 0.85 | 0.00 | 0.02 | 0.00 | 2472 | 195 | 49 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.00 | 0.02 | 0.70 | 2472 | 39 | 10 | 0/8 | 0.00 | 0.169 | 4 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Cat toy, Dog toy, Rex cat |
| meaning | 0.85 | 0.00 | 0.02 | 0.90 | 2472 | 167 | 46 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Domestic long-haired cat, Norwegian forest cat |
| meaning | 0.85 | 0.00 | 0.01 | 0.00 | 2472 | 456 | 103 | 0/8 | 0.00 | 0.169 | 34 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.00 | 0.01 | 0.70 | 2472 | 103 | 18 | 0/8 | 0.00 | 0.169 | 10 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Cat toy, Dog toy, Rex cat |
| meaning | 0.85 | 0.00 | 0.01 | 0.90 | 2472 | 404 | 100 | 0/8 | 0.00 | 0.169 | 35 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Kitten, Domestic long-haired cat, Norwegian forest cat |
| meaning | 0.85 | 0.30 | 0.02 | 0.00 | 2471 | 194 | 49 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.30 | 0.02 | 0.70 | 2471 | 39 | 10 | 0/8 | 0.00 | 0.169 | 4 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.85 | 0.30 | 0.02 | 0.90 | 2471 | 165 | 44 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Cat, Kitten, Domestic long-haired cat |
| meaning | 0.85 | 0.30 | 0.01 | 0.00 | 2471 | 456 | 105 | 0/8 | 0.00 | 0.169 | 34 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.30 | 0.01 | 0.70 | 2471 | 103 | 18 | 0/8 | 0.00 | 0.169 | 10 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.85 | 0.30 | 0.01 | 0.90 | 2471 | 403 | 100 | 0/8 | 0.00 | 0.169 | 35 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Cat, Kitten, Domestic long-haired cat |
| meaning | 0.85 | 0.50 | 0.02 | 0.00 | 2472 | 194 | 49 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.50 | 0.02 | 0.70 | 2472 | 39 | 10 | 0/8 | 0.00 | 0.169 | 4 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.85 | 0.50 | 0.02 | 0.90 | 2472 | 165 | 44 | 0/8 | 0.00 | 0.169 | 14 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Cat, Kitten, Domestic long-haired cat |
| meaning | 0.85 | 0.50 | 0.01 | 0.00 | 2472 | 455 | 104 | 0/8 | 0.00 | 0.169 | 34 | Bench | Bench (Geography) | Outdoor bench | American cocker spaniel, Cocker spaniel, English cocker spaniel, French spaniel, German spaniel, Picardy spaniel, Russian spaniel |
| meaning | 0.85 | 0.50 | 0.01 | 0.70 | 2472 | 102 | 17 | 0/8 | 0.00 | 0.169 | 10 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Cat bed, Dog bed, Rex cat, Tabby cat, Cat-ketch |
| meaning | 0.85 | 0.50 | 0.01 | 0.90 | 2472 | 402 | 99 | 0/8 | 0.00 | 0.169 | 35 | Bench | Bench (Geography) | Outdoor bench | Domestic short-haired cat, Polydactyl cat, Napoleon cat, Rex cat, Tabby cat, Cat, Kitten, Domestic long-haired cat |

## waterbirds, laion

| method | text | coact/resp | min | merge | groups | factors | composite | cross/pairs | precision | attr. signal | attr. factors | strongest attribute factor | largest factor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cospro | 0.60 | 0.10 | 0.01 | 0.00 | 133 | 133 | 9 | 0/8 | 0.00 | 0.122 | 10 | beaches | raven, crow |
| cospro | 0.60 | 0.10 | 0.01 | 0.90 | 133 | 131 | 11 | 0/8 | 0.00 | 0.122 | 9 | beaches | raven, crow |
| cospro | 0.60 | 0.10 | 0.005 | 0.00 | 133 | 133 | 9 | 0/8 | 0.00 | 0.122 | 10 | beaches | raven, crow |
| cospro | 0.60 | 0.10 | 0.005 | 0.90 | 133 | 131 | 11 | 0/8 | 0.00 | 0.122 | 9 | beaches | raven, crow |
| cospro | 0.60 | 0.30 | 0.01 | 0.00 | 137 | 137 | 5 | 0/8 | 0.00 | 0.122 | 10 | beaches | raven, crow |
| cospro | 0.60 | 0.30 | 0.01 | 0.90 | 137 | 134 | 8 | 0/8 | 0.00 | 0.122 | 9 | beaches | raven, crow |
| cospro | 0.60 | 0.30 | 0.005 | 0.00 | 137 | 137 | 5 | 0/8 | 0.00 | 0.122 | 10 | beaches | raven, crow |
| cospro | 0.60 | 0.30 | 0.005 | 0.90 | 137 | 134 | 8 | 0/8 | 0.00 | 0.122 | 9 | beaches | raven, crow |
| cospro | 0.80 | 0.10 | 0.01 | 0.00 | 137 | 137 | 5 | 0/8 | 0.00 | 0.122 | 10 | beaches | seagull, gull |
| cospro | 0.80 | 0.10 | 0.01 | 0.90 | 137 | 134 | 8 | 0/8 | 0.00 | 0.122 | 9 | beaches | crow, raven |
| cospro | 0.80 | 0.10 | 0.005 | 0.00 | 137 | 137 | 5 | 0/8 | 0.00 | 0.122 | 10 | beaches | seagull, gull |
| cospro | 0.80 | 0.10 | 0.005 | 0.90 | 137 | 134 | 8 | 0/8 | 0.00 | 0.122 | 9 | beaches | crow, raven |
| cospro | 0.80 | 0.30 | 0.01 | 0.00 | 138 | 138 | 4 | 0/8 | 0.00 | 0.122 | 10 | beaches | seagull, gull |
| cospro | 0.80 | 0.30 | 0.01 | 0.90 | 138 | 134 | 8 | 0/8 | 0.00 | 0.122 | 9 | beaches | crow, raven |
| cospro | 0.80 | 0.30 | 0.005 | 0.00 | 138 | 138 | 4 | 0/8 | 0.00 | 0.122 | 10 | beaches | seagull, gull |
| cospro | 0.80 | 0.30 | 0.005 | 0.90 | 138 | 134 | 8 | 0/8 | 0.00 | 0.122 | 9 | beaches | crow, raven |
| meaning | 0.85 | 0.00 | 0.01 | 0.00 | 407 | 139 | 53 | 0/8 | 0.00 | 0.122 | 12 | beaches | ss, java, innovative, summary, ---, newest, powerful, meme |
| meaning | 0.85 | 0.00 | 0.01 | 0.90 | 407 | 138 | 54 | 0/8 | 0.00 | 0.122 | 12 | beaches | ss, java, innovative, summary, ---, newest, powerful, meme |
| meaning | 0.85 | 0.00 | 0.005 | 0.00 | 407 | 232 | 75 | 0/8 | 0.00 | 0.122 | 20 | beaches | ss, java, innovative, summary, ---, newest, powerful, meme |
| meaning | 0.85 | 0.30 | 0.01 | 0.00 | 414 | 141 | 57 | 0/8 | 0.00 | 0.122 | 12 | beaches | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.30 | 0.01 | 0.90 | 414 | 140 | 58 | 0/8 | 0.00 | 0.122 | 12 | beaches | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.30 | 0.005 | 0.00 | 414 | 236 | 81 | 0/8 | 0.00 | 0.122 | 20 | beaches | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.50 | 0.01 | 0.00 | 428 | 143 | 57 | 0/8 | 0.00 | 0.122 | 11 | beaches | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.50 | 0.01 | 0.90 | 428 | 142 | 58 | 0/8 | 0.00 | 0.122 | 11 | beaches | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.50 | 0.005 | 0.00 | 428 | 239 | 80 | 0/8 | 0.00 | 0.122 | 19 | beaches | hikes, hikers, hiker, hike, hiking |
| meaning | 0.90 | 0.00 | 0.01 | 0.00 | 476 | 141 | 31 | 0/8 | 0.00 | 0.122 | 11 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.00 | 0.01 | 0.90 | 476 | 139 | 33 | 0/8 | 0.00 | 0.122 | 11 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.00 | 0.005 | 0.00 | 476 | 245 | 41 | 0/8 | 0.00 | 0.122 | 17 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.00 | 0.005 | 0.90 | 476 | 240 | 45 | 0/8 | 0.00 | 0.122 | 17 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.30 | 0.01 | 0.00 | 477 | 141 | 30 | 0/8 | 0.00 | 0.122 | 11 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.30 | 0.01 | 0.90 | 477 | 139 | 32 | 0/8 | 0.00 | 0.122 | 11 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.30 | 0.005 | 0.00 | 477 | 245 | 40 | 0/8 | 0.00 | 0.122 | 17 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.30 | 0.005 | 0.90 | 477 | 240 | 44 | 0/8 | 0.00 | 0.122 | 17 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.50 | 0.01 | 0.00 | 479 | 140 | 29 | 0/8 | 0.00 | 0.122 | 11 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.50 | 0.01 | 0.90 | 479 | 138 | 31 | 0/8 | 0.00 | 0.122 | 11 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.50 | 0.005 | 0.00 | 479 | 246 | 40 | 0/8 | 0.00 | 0.122 | 17 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.50 | 0.005 | 0.90 | 479 | 241 | 44 | 0/8 | 0.00 | 0.122 | 17 | beaches | reflects, reflecting, reflected, reflection |
| meaning | 0.80 | 0.00 | 0.01 | 0.00 | 297 | 122 | 69 | 0/8 | 0.00 | 0.120 | 7 | harbour | harbor | peep, poet, pi, ss, hen, java, torrent, supplied |
| meaning | 0.80 | 0.00 | 0.01 | 0.90 | 297 | 122 | 69 | 0/8 | 0.00 | 0.120 | 7 | harbour | harbor | peep, poet, pi, ss, hen, java, torrent, supplied |
| meaning | 0.80 | 0.00 | 0.005 | 0.00 | 297 | 187 | 91 | 0/8 | 0.00 | 0.120 | 14 | harbour | harbor | peep, poet, pi, ss, hen, java, torrent, supplied |
| meaning | 0.80 | 0.00 | 0.005 | 0.90 | 297 | 186 | 90 | 0/8 | 0.00 | 0.120 | 13 | harbour | harbor | peep, poet, pi, ss, hen, java, torrent, supplied |
| meaning | 0.80 | 0.30 | 0.01 | 0.00 | 326 | 137 | 84 | 0/8 | 0.00 | 0.120 | 8 | harbour | harbor | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.30 | 0.01 | 0.90 | 326 | 137 | 84 | 0/8 | 0.00 | 0.120 | 8 | harbour | harbor | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.30 | 0.005 | 0.00 | 326 | 209 | 112 | 0/8 | 0.00 | 0.120 | 16 | harbour | harbor | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.30 | 0.005 | 0.90 | 326 | 208 | 111 | 0/8 | 0.00 | 0.120 | 15 | harbour | harbor | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.50 | 0.01 | 0.00 | 359 | 137 | 77 | 0/8 | 0.00 | 0.120 | 8 | harbour | harbor | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.50 | 0.01 | 0.90 | 359 | 137 | 77 | 0/8 | 0.00 | 0.120 | 8 | harbour | harbor | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.50 | 0.005 | 0.00 | 359 | 221 | 108 | 0/8 | 0.00 | 0.120 | 15 | harbour | harbor | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.50 | 0.005 | 0.90 | 359 | 221 | 108 | 0/8 | 0.00 | 0.120 | 15 | harbour | harbor | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.85 | 0.00 | 0.005 | 0.90 | 407 | 229 | 76 | 0/8 | 0.00 | 0.120 | 19 | harbour | harbor | ss, java, innovative, summary, ---, newest, powerful, meme |
| meaning | 0.85 | 0.30 | 0.005 | 0.90 | 414 | 233 | 82 | 0/8 | 0.00 | 0.120 | 19 | harbour | harbor | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.50 | 0.005 | 0.90 | 428 | 236 | 81 | 0/8 | 0.00 | 0.120 | 18 | harbour | harbor | hikes, hikers, hiker, hike, hiking |
| cospro | 0.60 | 0.10 | 0.02 | 0.00 | 133 | 71 | 9 | 0/8 | 0.00 | 0.115 | 4 | seascape | raven, crow |
| cospro | 0.60 | 0.10 | 0.02 | 0.90 | 133 | 70 | 10 | 0/8 | 0.00 | 0.115 | 4 | seascape | raven, crow |
| cospro | 0.60 | 0.30 | 0.02 | 0.00 | 137 | 74 | 5 | 0/8 | 0.00 | 0.115 | 3 | seascape | raven, crow |
| cospro | 0.60 | 0.30 | 0.02 | 0.90 | 137 | 72 | 7 | 0/8 | 0.00 | 0.115 | 3 | seascape | raven, crow |
| cospro | 0.80 | 0.10 | 0.02 | 0.00 | 137 | 74 | 5 | 0/8 | 0.00 | 0.115 | 3 | seascape | seagull, gull |
| cospro | 0.80 | 0.10 | 0.02 | 0.90 | 137 | 72 | 7 | 0/8 | 0.00 | 0.115 | 3 | seascape | crow, raven |
| cospro | 0.80 | 0.30 | 0.02 | 0.00 | 138 | 75 | 4 | 0/8 | 0.00 | 0.115 | 3 | seascape | seagull, gull |
| cospro | 0.80 | 0.30 | 0.02 | 0.90 | 138 | 72 | 7 | 0/8 | 0.00 | 0.115 | 3 | seascape | crow, raven |
| meaning | 0.75 | 0.00 | 0.02 | 0.00 | 196 | 50 | 29 | 0/7 | 0.00 | 0.115 | 4 | seascape | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| meaning | 0.75 | 0.00 | 0.02 | 0.90 | 196 | 50 | 29 | 0/7 | 0.00 | 0.115 | 4 | seascape | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| meaning | 0.75 | 0.00 | 0.01 | 0.00 | 196 | 89 | 54 | 0/8 | 0.00 | 0.115 | 5 | seascape | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| meaning | 0.75 | 0.00 | 0.01 | 0.90 | 196 | 89 | 54 | 0/8 | 0.00 | 0.115 | 5 | seascape | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| meaning | 0.75 | 0.00 | 0.005 | 0.00 | 196 | 129 | 71 | 0/8 | 0.00 | 0.115 | 8 | seascape | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| meaning | 0.75 | 0.00 | 0.005 | 0.90 | 196 | 129 | 71 | 0/8 | 0.00 | 0.115 | 8 | seascape | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| meaning | 0.75 | 0.30 | 0.02 | 0.00 | 253 | 76 | 58 | 0/8 | 0.00 | 0.115 | 7 | seascape | swans, geese, pelican, goose, ducks, flamingo, swan, penguins |
| meaning | 0.75 | 0.30 | 0.02 | 0.90 | 253 | 76 | 58 | 0/8 | 0.00 | 0.115 | 7 | seascape | swans, geese, pelican, goose, ducks, flamingo, swan, penguins |
| meaning | 0.75 | 0.30 | 0.01 | 0.00 | 253 | 127 | 93 | 0/8 | 0.00 | 0.115 | 9 | seascape | swans, geese, pelican, goose, ducks, flamingo, swan, penguins |
| meaning | 0.75 | 0.30 | 0.01 | 0.90 | 253 | 127 | 93 | 0/8 | 0.00 | 0.115 | 9 | seascape | swans, geese, pelican, goose, ducks, flamingo, swan, penguins |
| meaning | 0.75 | 0.30 | 0.005 | 0.00 | 253 | 176 | 116 | 0/8 | 0.00 | 0.115 | 12 | seascape | swans, geese, pelican, goose, ducks, flamingo, swan, penguins |
| meaning | 0.75 | 0.30 | 0.005 | 0.90 | 253 | 176 | 116 | 0/8 | 0.00 | 0.115 | 12 | seascape | swans, geese, pelican, goose, ducks, flamingo, swan, penguins |
| meaning | 0.75 | 0.50 | 0.02 | 0.00 | 315 | 76 | 53 | 0/8 | 0.00 | 0.115 | 7 | seascape | boating, cruises, maritime, charter, shark, yacht, sailing, cruise |
| meaning | 0.75 | 0.50 | 0.02 | 0.90 | 315 | 76 | 53 | 0/8 | 0.00 | 0.115 | 7 | seascape | boating, cruises, maritime, charter, shark, yacht, sailing, cruise |
| meaning | 0.75 | 0.50 | 0.01 | 0.00 | 315 | 138 | 91 | 0/8 | 0.00 | 0.115 | 9 | seascape | boating, cruises, maritime, charter, shark, yacht, sailing, cruise |
| meaning | 0.75 | 0.50 | 0.01 | 0.90 | 315 | 138 | 91 | 0/8 | 0.00 | 0.115 | 9 | seascape | boating, cruises, maritime, charter, shark, yacht, sailing, cruise |
| meaning | 0.75 | 0.50 | 0.005 | 0.00 | 315 | 205 | 116 | 0/8 | 0.00 | 0.115 | 13 | seascape | boating, cruises, maritime, charter, shark, yacht, sailing, cruise |
| meaning | 0.75 | 0.50 | 0.005 | 0.90 | 315 | 205 | 116 | 0/8 | 0.00 | 0.115 | 13 | seascape | boating, cruises, maritime, charter, shark, yacht, sailing, cruise |
| meaning | 0.80 | 0.00 | 0.02 | 0.00 | 297 | 76 | 47 | 0/8 | 0.00 | 0.115 | 5 | seascape | peep, poet, pi, ss, hen, java, torrent, supplied |
| meaning | 0.80 | 0.00 | 0.02 | 0.90 | 297 | 76 | 47 | 0/8 | 0.00 | 0.115 | 5 | seascape | peep, poet, pi, ss, hen, java, torrent, supplied |
| meaning | 0.80 | 0.30 | 0.02 | 0.00 | 326 | 83 | 52 | 0/8 | 0.00 | 0.115 | 5 | seascape | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.30 | 0.02 | 0.90 | 326 | 83 | 52 | 0/8 | 0.00 | 0.115 | 5 | seascape | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.50 | 0.02 | 0.00 | 359 | 76 | 44 | 0/8 | 0.00 | 0.115 | 5 | seascape | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.80 | 0.50 | 0.02 | 0.90 | 359 | 76 | 44 | 0/8 | 0.00 | 0.115 | 5 | seascape | hikes, hikers, hiker, trekking, hike, hiking |
| meaning | 0.85 | 0.00 | 0.02 | 0.00 | 407 | 77 | 32 | 0/8 | 0.00 | 0.115 | 3 | seascape | ss, java, innovative, summary, ---, newest, powerful, meme |
| meaning | 0.85 | 0.00 | 0.02 | 0.90 | 407 | 76 | 33 | 0/8 | 0.00 | 0.115 | 3 | seascape | ss, java, innovative, summary, ---, newest, powerful, meme |
| meaning | 0.85 | 0.30 | 0.02 | 0.00 | 414 | 78 | 33 | 0/8 | 0.00 | 0.115 | 3 | seascape | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.30 | 0.02 | 0.90 | 414 | 77 | 34 | 0/8 | 0.00 | 0.115 | 3 | seascape | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.50 | 0.02 | 0.00 | 428 | 74 | 29 | 0/8 | 0.00 | 0.115 | 3 | seascape | hikes, hikers, hiker, hike, hiking |
| meaning | 0.85 | 0.50 | 0.02 | 0.90 | 428 | 73 | 30 | 0/8 | 0.00 | 0.115 | 3 | seascape | hikes, hikers, hiker, hike, hiking |
| meaning | 0.90 | 0.00 | 0.02 | 0.00 | 476 | 74 | 18 | 0/8 | 0.00 | 0.115 | 3 | seascape | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.00 | 0.02 | 0.90 | 476 | 72 | 20 | 0/8 | 0.00 | 0.115 | 3 | seascape | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.30 | 0.02 | 0.00 | 477 | 74 | 17 | 0/8 | 0.00 | 0.115 | 3 | seascape | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.30 | 0.02 | 0.90 | 477 | 72 | 19 | 0/8 | 0.00 | 0.115 | 3 | seascape | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.50 | 0.02 | 0.00 | 479 | 74 | 17 | 0/8 | 0.00 | 0.115 | 3 | seascape | reflects, reflecting, reflected, reflection |
| meaning | 0.90 | 0.50 | 0.02 | 0.90 | 479 | 72 | 19 | 0/8 | 0.00 | 0.115 | 3 | seascape | reflects, reflecting, reflected, reflection |
| meaning | 0.80 | 0.50 | 0.01 | 0.70 | 359 | 116 | 62 | 0/8 | 0.00 | 0.106 | 3 | whales | dolphins | dolphin (+19) | whales, dolphins, dolphin, whale, coastline, seaside, beaches, beach |
| meaning | 0.80 | 0.30 | 0.02 | 0.70 | 326 | 74 | 47 | 0/8 | 0.00 | 0.099 | 3 | whales | dolphins | dolphin (+6) | forests, rainforest, forest, woodland, woods, herbal, herb, maple |
| meaning | 0.80 | 0.50 | 0.02 | 0.70 | 359 | 68 | 40 | 0/8 | 0.00 | 0.099 | 3 | whales | dolphins | dolphin (+6) | whales, dolphins, dolphin, whale, coastline, seaside, beaches, beach |
| meaning | 0.75 | 0.50 | 0.005 | 0.70 | 315 | 162 | 90 | 0/8 | 0.00 | 0.095 | 5 | paradise | islands | island | falling, floating, flying, whales, dolphins, dolphin, whale, fishermen |
| meaning | 0.80 | 0.30 | 0.005 | 0.70 | 326 | 160 | 86 | 0/8 | 0.00 | 0.095 | 5 | paradise | islands | island | falling, floating, flying, whales, dolphins, dolphin, whale, fishermen |
| meaning | 0.80 | 0.50 | 0.005 | 0.70 | 359 | 167 | 80 | 0/8 | 0.00 | 0.095 | 5 | paradise | islands | island | falling, floating, flying, whales, dolphins, dolphin, whale, fishermen |
| meaning | 0.75 | 0.00 | 0.005 | 0.70 | 196 | 106 | 61 | 0/8 | 0.00 | 0.093 | 3 | tsunami | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| meaning | 0.75 | 0.30 | 0.005 | 0.70 | 253 | 146 | 100 | 0/8 | 0.00 | 0.093 | 4 | tsunami | sinking, falling, floating, flying, whales, dolphins, dolphin, whale |
| meaning | 0.80 | 0.00 | 0.005 | 0.70 | 297 | 142 | 67 | 0/8 | 0.00 | 0.093 | 4 | tsunami | peep, poet, pi, ss, hen, java, torrent, supplied |
| meaning | 0.75 | 0.00 | 0.01 | 0.70 | 196 | 76 | 46 | 0/6 | 0.00 | 0.091 | 1 | whales | dolphins | dolphin (+21) | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| meaning | 0.75 | 0.00 | 0.02 | 0.70 | 196 | 44 | 25 | 0/6 | 0.00 | 0.090 | 1 | whales | dolphins | dolphin (+14) | identifying, vr, alaska, tit, wc, mandarin, enchanting, mammal |
| cospro | 0.60 | 0.10 | 0.01 | 0.70 | 133 | 105 | 17 | 0/8 | 0.00 | 0.078 | 6 | lake | lakes | duck, pelican, dolphin, whale, goose, seascape, seal, beaches |
| cospro | 0.60 | 0.10 | 0.005 | 0.70 | 133 | 105 | 17 | 0/8 | 0.00 | 0.078 | 6 | lake | lakes | duck, pelican, dolphin, whale, goose, seascape, seal, beaches |
| cospro | 0.60 | 0.30 | 0.01 | 0.70 | 137 | 106 | 18 | 0/8 | 0.00 | 0.078 | 6 | lake | lakes | duck, pelican, goose, whale, seascape, seal, beaches, dolphin |
| cospro | 0.60 | 0.30 | 0.005 | 0.70 | 137 | 106 | 18 | 0/8 | 0.00 | 0.078 | 6 | lake | lakes | duck, pelican, goose, whale, seascape, seal, beaches, dolphin |
| cospro | 0.80 | 0.10 | 0.01 | 0.70 | 137 | 106 | 18 | 0/8 | 0.00 | 0.078 | 6 | lake | lakes | duck, pelican, goose, whale, seascape, seal, beaches, dolphin |
| cospro | 0.80 | 0.10 | 0.005 | 0.70 | 137 | 106 | 18 | 0/8 | 0.00 | 0.078 | 6 | lake | lakes | duck, pelican, goose, whale, seascape, seal, beaches, dolphin |
| cospro | 0.80 | 0.30 | 0.01 | 0.70 | 138 | 106 | 18 | 0/8 | 0.00 | 0.078 | 6 | lake | lakes | duck, pelican, goose, whale, seascape, seal, beaches, dolphin |
| cospro | 0.80 | 0.30 | 0.005 | 0.70 | 138 | 106 | 18 | 0/8 | 0.00 | 0.078 | 6 | lake | lakes | duck, pelican, goose, whale, seascape, seal, beaches, dolphin |
| meaning | 0.85 | 0.00 | 0.02 | 0.70 | 407 | 64 | 30 | 0/8 | 0.00 | 0.078 | 3 | lakes | lake | ss, java, innovative, summary, ---, newest, powerful, meme |
| meaning | 0.85 | 0.00 | 0.01 | 0.70 | 407 | 113 | 44 | 0/8 | 0.00 | 0.078 | 6 | lakes | lake | ducks, duck, pelican, geese, goose, whales, whale, seascape |
| meaning | 0.85 | 0.00 | 0.005 | 0.70 | 407 | 158 | 57 | 0/8 | 0.00 | 0.078 | 4 | lakes | lake | ducks, duck, pelican, geese, goose, fishing, fish, whales |
| meaning | 0.85 | 0.30 | 0.02 | 0.70 | 414 | 65 | 31 | 0/8 | 0.00 | 0.078 | 3 | lakes | lake | forests, forest, rainforest, woodland, maple, nature, tree, branches |
| meaning | 0.85 | 0.30 | 0.01 | 0.70 | 414 | 115 | 48 | 0/8 | 0.00 | 0.078 | 6 | lakes | lake | ducks, duck, pelican, geese, goose, whales, whale, seascape |
| meaning | 0.85 | 0.30 | 0.005 | 0.70 | 414 | 162 | 63 | 0/8 | 0.00 | 0.078 | 4 | lakes | lake | ducks, duck, pelican, geese, goose, fishing, fish, whales |
| meaning | 0.85 | 0.50 | 0.02 | 0.70 | 428 | 61 | 27 | 0/8 | 0.00 | 0.078 | 3 | lakes | lake | forests, forest, rainforest, woodland, nature, tree, branches, branch |
| meaning | 0.85 | 0.50 | 0.01 | 0.70 | 428 | 117 | 48 | 0/8 | 0.00 | 0.078 | 5 | lakes | lake | ducks, duck, pelican, geese, goose, whales, whale, seascape |
| meaning | 0.85 | 0.50 | 0.005 | 0.70 | 428 | 163 | 60 | 0/8 | 0.00 | 0.078 | 4 | lakes | lake | ducks, duck, pelican, geese, goose, fishing, fish, whales |
| meaning | 0.90 | 0.00 | 0.02 | 0.70 | 476 | 58 | 22 | 0/8 | 0.00 | 0.078 | 3 | lakes | lake | forests, forest, rainforest, woodland, hikes, hike, hiking, branch |
| meaning | 0.90 | 0.00 | 0.01 | 0.70 | 476 | 112 | 30 | 0/8 | 0.00 | 0.078 | 6 | lakes | lake | ducks, duck, pelican, geese, goose, whales, whale, seascape |
| meaning | 0.90 | 0.00 | 0.005 | 0.70 | 476 | 169 | 44 | 0/8 | 0.00 | 0.078 | 5 | lakes | lake | forests, forest, rainforest, woodland, hikes, hike, hiking, marsh |
| meaning | 0.90 | 0.30 | 0.02 | 0.70 | 477 | 57 | 22 | 0/8 | 0.00 | 0.078 | 3 | lakes | lake | forests, forest, rainforest, woodland, hikes, hike, hiking, branch |
| meaning | 0.90 | 0.30 | 0.01 | 0.70 | 477 | 111 | 30 | 0/8 | 0.00 | 0.078 | 6 | lakes | lake | ducks, duck, pelican, geese, goose, whales, whale, seascape |
| meaning | 0.90 | 0.30 | 0.005 | 0.70 | 477 | 168 | 44 | 0/8 | 0.00 | 0.078 | 5 | lakes | lake | forests, forest, rainforest, woodland, hikes, hike, hiking, marsh |
| meaning | 0.90 | 0.50 | 0.02 | 0.70 | 479 | 57 | 22 | 0/8 | 0.00 | 0.078 | 3 | lakes | lake | forests, forest, rainforest, woodland, hikes, hike, hiking, branch |
| meaning | 0.90 | 0.50 | 0.01 | 0.70 | 479 | 110 | 29 | 0/8 | 0.00 | 0.078 | 6 | lakes | lake | ducks, duck, pelican, geese, goose, whales, whale, seascape |
| meaning | 0.90 | 0.50 | 0.005 | 0.70 | 479 | 169 | 44 | 0/8 | 0.00 | 0.078 | 5 | lakes | lake | forests, forest, rainforest, woodland, hikes, hike, hiking, marsh |
| meaning | 0.80 | 0.30 | 0.01 | 0.70 | 326 | 116 | 70 | 0/8 | 0.00 | 0.078 | 3 | whales | dolphins | dolphin (+20) | whales, dolphins, dolphin, whale, coastline, seaside, beaches, beach |
| cospro | 0.60 | 0.10 | 0.02 | 0.70 | 133 | 52 | 15 | 0/8 | 0.00 | 0.076 | 2 | lake | forests, forest, rainforest, woodland, hiking, branch, tree, jungle |
| cospro | 0.60 | 0.30 | 0.02 | 0.70 | 137 | 54 | 17 | 0/8 | 0.00 | 0.076 | 3 | lake | forests, forest, rainforest, woodland, hiking, branch, tree, jungle |
| cospro | 0.80 | 0.10 | 0.02 | 0.70 | 137 | 54 | 17 | 0/8 | 0.00 | 0.076 | 3 | lake | forests, forest, rainforest, woodland, hiking, branch, tree, jungle |
| cospro | 0.80 | 0.30 | 0.02 | 0.70 | 138 | 54 | 17 | 0/8 | 0.00 | 0.076 | 3 | lake | forests, forest, rainforest, woodland, hiking, branch, tree, jungle |
| meaning | 0.75 | 0.30 | 0.02 | 0.70 | 253 | 66 | 50 | 0/8 | 0.00 | 0.074 | 2 | lakes | waters | lake | sinking, falling, floating, flying, whales, dolphins, dolphin, whale |
| meaning | 0.75 | 0.30 | 0.01 | 0.70 | 253 | 109 | 80 | 0/8 | 0.00 | 0.074 | 3 | lakes | waters | lake | sinking, falling, floating, flying, whales, dolphins, dolphin, whale |
| meaning | 0.75 | 0.50 | 0.02 | 0.70 | 315 | 63 | 42 | 0/8 | 0.00 | 0.074 | 2 | lakes | waters | lake | falling, floating, flying, whales, dolphins, dolphin, whale, fishermen |
| meaning | 0.75 | 0.50 | 0.01 | 0.70 | 315 | 112 | 70 | 0/8 | 0.00 | 0.074 | 3 | lakes | waters | lake | falling, floating, flying, whales, dolphins, dolphin, whale, fishermen |
| meaning | 0.80 | 0.00 | 0.02 | 0.70 | 297 | 65 | 40 | 0/8 | 0.00 | 0.055 | 1 | dawn | glowing | sunrise (+1) | peep, poet, pi, ss, hen, java, torrent, supplied |
| meaning | 0.80 | 0.00 | 0.01 | 0.70 | 297 | 104 | 58 | 0/8 | 0.00 | 0.055 | 1 | dawn | glowing | sunrise (+1) | peep, poet, pi, ss, hen, java, torrent, supplied |

## waterbirds, openimages_v7

| method | text | coact/resp | min | merge | groups | factors | composite | cross/pairs | precision | attr. signal | attr. factors | strongest attribute factor | largest factor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cospro | 0.60 | 0.10 | 0.02 | 0.70 | 431 | 108 | 20 | 1/8 | 0.12 | 0.060 | 3 | Pomacentridae | Bird feeder, Bird food, Perching bird, American Tree Sparrow, Bewick s Wren, Bushtit, Field Sparrow, Fox Sparrow |
| meaning | 0.80 | 0.00 | 0.02 | 0.70 | 1481 | 104 | 35 | 1/8 | 0.12 | 0.060 | 3 | Pomacentridae | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.80 | 0.30 | 0.02 | 0.70 | 1482 | 104 | 35 | 1/8 | 0.12 | 0.060 | 3 | Pomacentridae | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.80 | 0.50 | 0.02 | 0.70 | 1482 | 104 | 35 | 1/8 | 0.12 | 0.060 | 3 | Pomacentridae | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.85 | 0.00 | 0.02 | 0.70 | 1633 | 106 | 27 | 1/8 | 0.12 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| meaning | 0.85 | 0.30 | 0.02 | 0.70 | 1633 | 106 | 27 | 1/8 | 0.12 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| meaning | 0.85 | 0.50 | 0.02 | 0.70 | 1633 | 106 | 27 | 1/8 | 0.12 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| meaning | 0.75 | 0.00 | 0.005 | 0.00 | 1257 | 732 | 267 | 0/8 | 0.00 | 0.132 | 32 | Singing sand | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.00 | 0.005 | 0.70 | 1257 | 340 | 105 | 0/8 | 0.00 | 0.132 | 11 | Singing sand | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.75 | 0.00 | 0.005 | 0.90 | 1257 | 716 | 254 | 0/8 | 0.00 | 0.132 | 32 | Singing sand | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.30 | 0.005 | 0.00 | 1261 | 735 | 269 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.30 | 0.005 | 0.70 | 1261 | 334 | 105 | 0/8 | 0.00 | 0.132 | 11 | Singing sand | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.75 | 0.30 | 0.005 | 0.90 | 1261 | 719 | 256 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.50 | 0.005 | 0.00 | 1281 | 745 | 265 | 0/8 | 0.00 | 0.132 | 34 | Singing sand | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.50 | 0.005 | 0.70 | 1281 | 340 | 97 | 0/8 | 0.00 | 0.132 | 12 | Singing sand | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.75 | 0.50 | 0.005 | 0.90 | 1281 | 729 | 252 | 0/8 | 0.00 | 0.132 | 34 | Singing sand | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.80 | 0.00 | 0.005 | 0.00 | 1481 | 829 | 197 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.00 | 0.005 | 0.70 | 1481 | 344 | 75 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.80 | 0.00 | 0.005 | 0.90 | 1481 | 801 | 183 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.30 | 0.005 | 0.00 | 1482 | 830 | 197 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.30 | 0.005 | 0.70 | 1482 | 344 | 76 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.80 | 0.30 | 0.005 | 0.90 | 1482 | 802 | 183 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.50 | 0.005 | 0.00 | 1482 | 830 | 197 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.50 | 0.005 | 0.70 | 1482 | 344 | 76 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.80 | 0.50 | 0.005 | 0.90 | 1482 | 802 | 183 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.85 | 0.00 | 0.005 | 0.00 | 1633 | 887 | 122 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.00 | 0.005 | 0.70 | 1633 | 340 | 58 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Dockrillia striolata |
| meaning | 0.85 | 0.00 | 0.005 | 0.90 | 1633 | 842 | 118 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.85 | 0.30 | 0.005 | 0.00 | 1633 | 887 | 122 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.30 | 0.005 | 0.70 | 1633 | 340 | 58 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Dockrillia striolata |
| meaning | 0.85 | 0.30 | 0.005 | 0.90 | 1633 | 842 | 118 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.85 | 0.50 | 0.005 | 0.00 | 1633 | 887 | 122 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.50 | 0.005 | 0.70 | 1633 | 340 | 58 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Dockrillia striolata |
| meaning | 0.85 | 0.50 | 0.005 | 0.90 | 1633 | 842 | 118 | 0/8 | 0.00 | 0.132 | 33 | Singing sand | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.90 | 0.00 | 0.005 | 0.00 | 1736 | 911 | 37 | 0/8 | 0.00 | 0.132 | 31 | Singing sand | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.00 | 0.005 | 0.70 | 1736 | 337 | 40 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Hesperornis, Nest box, Forest |
| meaning | 0.90 | 0.00 | 0.005 | 0.90 | 1736 | 841 | 61 | 0/8 | 0.00 | 0.132 | 31 | Singing sand | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| meaning | 0.90 | 0.30 | 0.005 | 0.00 | 1736 | 911 | 37 | 0/8 | 0.00 | 0.132 | 31 | Singing sand | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.30 | 0.005 | 0.70 | 1736 | 337 | 40 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Hesperornis, Nest box, Forest |
| meaning | 0.90 | 0.30 | 0.005 | 0.90 | 1736 | 841 | 61 | 0/8 | 0.00 | 0.132 | 31 | Singing sand | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| meaning | 0.90 | 0.50 | 0.005 | 0.00 | 1736 | 911 | 37 | 0/8 | 0.00 | 0.132 | 31 | Singing sand | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.50 | 0.005 | 0.70 | 1736 | 337 | 40 | 0/8 | 0.00 | 0.132 | 13 | Singing sand | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Hesperornis, Nest box, Forest |
| meaning | 0.90 | 0.50 | 0.005 | 0.90 | 1736 | 841 | 61 | 0/8 | 0.00 | 0.132 | 31 | Singing sand | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.60 | 0.10 | 0.01 | 0.00 | 431 | 431 | 36 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American Tree Sparrow, Bewick s Wren, Bushtit, Field Sparrow, Fox Sparrow, House Wren, House sparrow, Marsh Wren |
| cospro | 0.60 | 0.10 | 0.01 | 0.90 | 431 | 414 | 42 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American Tree Sparrow, Bewick s Wren, Bushtit, Field Sparrow, Fox Sparrow, House Wren, House sparrow, Marsh Wren |
| cospro | 0.60 | 0.10 | 0.005 | 0.00 | 431 | 431 | 36 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American Tree Sparrow, Bewick s Wren, Bushtit, Field Sparrow, Fox Sparrow, House Wren, House sparrow, Marsh Wren |
| cospro | 0.60 | 0.10 | 0.005 | 0.90 | 431 | 414 | 42 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American Tree Sparrow, Bewick s Wren, Bushtit, Field Sparrow, Fox Sparrow, House Wren, House sparrow, Marsh Wren |
| cospro | 0.60 | 0.30 | 0.01 | 0.00 | 490 | 490 | 10 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Common Raven, Crow, Crow-like bird, Fish Crow, New caledonian crow, Raven |
| cospro | 0.60 | 0.30 | 0.01 | 0.90 | 490 | 444 | 29 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.60 | 0.30 | 0.005 | 0.00 | 490 | 490 | 10 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Common Raven, Crow, Crow-like bird, Fish Crow, New caledonian crow, Raven |
| cospro | 0.60 | 0.30 | 0.005 | 0.90 | 490 | 444 | 29 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.80 | 0.10 | 0.01 | 0.00 | 495 | 495 | 10 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American Tree Sparrow, Field Sparrow |
| cospro | 0.80 | 0.10 | 0.01 | 0.90 | 495 | 445 | 27 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.80 | 0.10 | 0.005 | 0.00 | 495 | 495 | 10 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American Tree Sparrow, Field Sparrow |
| cospro | 0.80 | 0.10 | 0.005 | 0.90 | 495 | 445 | 27 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.80 | 0.30 | 0.01 | 0.00 | 502 | 502 | 3 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Cormorant, Double crested Cormorant |
| cospro | 0.80 | 0.30 | 0.01 | 0.90 | 502 | 446 | 28 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.80 | 0.30 | 0.005 | 0.00 | 502 | 502 | 3 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Cormorant, Double crested Cormorant |
| cospro | 0.80 | 0.30 | 0.005 | 0.90 | 502 | 446 | 28 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| meaning | 0.80 | 0.00 | 0.01 | 0.00 | 1481 | 462 | 143 | 0/8 | 0.00 | 0.123 | 12 | Surf fishing | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.00 | 0.01 | 0.90 | 1481 | 438 | 129 | 0/8 | 0.00 | 0.123 | 12 | Surf fishing | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.30 | 0.01 | 0.00 | 1482 | 463 | 143 | 0/8 | 0.00 | 0.123 | 12 | Surf fishing | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.30 | 0.01 | 0.90 | 1482 | 439 | 129 | 0/8 | 0.00 | 0.123 | 12 | Surf fishing | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.50 | 0.01 | 0.00 | 1482 | 463 | 143 | 0/8 | 0.00 | 0.123 | 12 | Surf fishing | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.50 | 0.01 | 0.90 | 1482 | 439 | 129 | 0/8 | 0.00 | 0.123 | 12 | Surf fishing | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.85 | 0.00 | 0.01 | 0.00 | 1633 | 487 | 86 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.00 | 0.01 | 0.90 | 1633 | 449 | 81 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.85 | 0.30 | 0.01 | 0.00 | 1633 | 487 | 86 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.30 | 0.01 | 0.90 | 1633 | 449 | 81 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.85 | 0.50 | 0.01 | 0.00 | 1633 | 487 | 86 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.50 | 0.01 | 0.90 | 1633 | 449 | 81 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.90 | 0.00 | 0.01 | 0.00 | 1736 | 500 | 27 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.00 | 0.01 | 0.90 | 1736 | 448 | 44 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| meaning | 0.90 | 0.30 | 0.01 | 0.00 | 1736 | 500 | 27 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.30 | 0.01 | 0.90 | 1736 | 448 | 44 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| meaning | 0.90 | 0.50 | 0.01 | 0.00 | 1736 | 500 | 27 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.50 | 0.01 | 0.90 | 1736 | 448 | 44 | 0/8 | 0.00 | 0.123 | 13 | Surf fishing | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.60 | 0.10 | 0.01 | 0.70 | 431 | 188 | 23 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Chrysopogon zizanioides, Dorotheanthus bellidiformis, Guizotia abyssinica, Microseris lanceolata, Pholisora catullus, Bird feeder, Bird food |
| cospro | 0.60 | 0.10 | 0.005 | 0.70 | 431 | 188 | 23 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Chrysopogon zizanioides, Dorotheanthus bellidiformis, Guizotia abyssinica, Microseris lanceolata, Pholisora catullus, Bird feeder, Bird food |
| cospro | 0.60 | 0.30 | 0.01 | 0.70 | 490 | 185 | 21 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| cospro | 0.60 | 0.30 | 0.005 | 0.70 | 490 | 185 | 21 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| cospro | 0.80 | 0.10 | 0.01 | 0.70 | 495 | 184 | 20 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| cospro | 0.80 | 0.10 | 0.005 | 0.70 | 495 | 184 | 20 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| cospro | 0.80 | 0.30 | 0.01 | 0.70 | 502 | 184 | 20 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| cospro | 0.80 | 0.30 | 0.005 | 0.70 | 502 | 184 | 20 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| meaning | 0.90 | 0.00 | 0.01 | 0.70 | 1736 | 184 | 23 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| meaning | 0.90 | 0.30 | 0.01 | 0.70 | 1736 | 184 | 23 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| meaning | 0.90 | 0.50 | 0.01 | 0.70 | 1736 | 184 | 23 | 0/8 | 0.00 | 0.096 | 11 | Surf fishing | Sex on the beach | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Acridotheres |
| meaning | 0.75 | 0.00 | 0.01 | 0.00 | 1257 | 427 | 197 | 0/8 | 0.00 | 0.095 | 12 | Cliff jumping | Dock jumping | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.00 | 0.01 | 0.70 | 1257 | 188 | 77 | 0/8 | 0.00 | 0.095 | 9 | Cliff jumping | Dock jumping | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.75 | 0.00 | 0.01 | 0.90 | 1257 | 412 | 185 | 0/8 | 0.00 | 0.095 | 12 | Cliff jumping | Dock jumping | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.30 | 0.01 | 0.00 | 1261 | 429 | 199 | 0/8 | 0.00 | 0.095 | 12 | Cliff jumping | Dock jumping | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.30 | 0.01 | 0.70 | 1261 | 185 | 78 | 0/8 | 0.00 | 0.095 | 9 | Cliff jumping | Dock jumping | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.75 | 0.30 | 0.01 | 0.90 | 1261 | 414 | 187 | 0/8 | 0.00 | 0.095 | 12 | Cliff jumping | Dock jumping | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.50 | 0.01 | 0.00 | 1281 | 432 | 195 | 0/8 | 0.00 | 0.095 | 14 | Cliff jumping | Dock jumping | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.50 | 0.01 | 0.70 | 1281 | 186 | 73 | 0/8 | 0.00 | 0.095 | 10 | Cliff jumping | Dock jumping | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.75 | 0.50 | 0.01 | 0.90 | 1281 | 417 | 183 | 0/8 | 0.00 | 0.095 | 14 | Cliff jumping | Dock jumping | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.00 | 0.02 | 0.00 | 1257 | 231 | 119 | 0/8 | 0.00 | 0.094 | 5 | Rock fishing | Surf fishing | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.00 | 0.02 | 0.90 | 1257 | 219 | 109 | 0/8 | 0.00 | 0.094 | 5 | Rock fishing | Surf fishing | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.30 | 0.02 | 0.00 | 1261 | 233 | 119 | 0/8 | 0.00 | 0.094 | 5 | Rock fishing | Surf fishing | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.30 | 0.02 | 0.90 | 1261 | 221 | 109 | 0/8 | 0.00 | 0.094 | 5 | Rock fishing | Surf fishing | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.50 | 0.02 | 0.00 | 1281 | 233 | 117 | 0/8 | 0.00 | 0.094 | 5 | Rock fishing | Surf fishing | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.75 | 0.50 | 0.02 | 0.90 | 1281 | 221 | 107 | 0/8 | 0.00 | 0.094 | 5 | Rock fishing | Surf fishing | Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus, Chrysobalanus icaco |
| meaning | 0.80 | 0.00 | 0.01 | 0.70 | 1481 | 185 | 52 | 0/8 | 0.00 | 0.091 | 8 | Cliff jumping | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.80 | 0.30 | 0.01 | 0.70 | 1482 | 186 | 53 | 0/8 | 0.00 | 0.091 | 8 | Cliff jumping | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.80 | 0.50 | 0.01 | 0.70 | 1482 | 186 | 53 | 0/8 | 0.00 | 0.091 | 8 | Cliff jumping | Vulnerable Native Breeds, Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata |
| meaning | 0.85 | 0.00 | 0.01 | 0.70 | 1633 | 187 | 37 | 0/8 | 0.00 | 0.091 | 11 | Cliff jumping | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Dockrillia striolata |
| meaning | 0.85 | 0.30 | 0.01 | 0.70 | 1633 | 187 | 37 | 0/8 | 0.00 | 0.091 | 11 | Cliff jumping | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Dockrillia striolata |
| meaning | 0.85 | 0.50 | 0.01 | 0.70 | 1633 | 187 | 37 | 0/8 | 0.00 | 0.091 | 11 | Cliff jumping | Vulnerable Native Breeds, Pseudemys concinna concinna, Bamboo, Perching bird, Bird feeder, Nest box, Forest, Dockrillia striolata |
| cospro | 0.60 | 0.10 | 0.02 | 0.00 | 431 | 220 | 35 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | American Tree Sparrow, Bewick s Wren, Bushtit, Field Sparrow, Fox Sparrow, House Wren, House sparrow, Marsh Wren |
| cospro | 0.60 | 0.10 | 0.02 | 0.90 | 431 | 211 | 38 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | American Tree Sparrow, Bewick s Wren, Bushtit, Field Sparrow, Fox Sparrow, House Wren, House sparrow, Marsh Wren |
| cospro | 0.60 | 0.30 | 0.02 | 0.00 | 490 | 234 | 10 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Common Raven, Crow, Crow-like bird, Fish Crow, New caledonian crow, Raven |
| cospro | 0.60 | 0.30 | 0.02 | 0.70 | 490 | 107 | 14 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| cospro | 0.60 | 0.30 | 0.02 | 0.90 | 490 | 217 | 16 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.80 | 0.10 | 0.02 | 0.00 | 495 | 238 | 10 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | American Tree Sparrow, Field Sparrow |
| cospro | 0.80 | 0.10 | 0.02 | 0.70 | 495 | 107 | 15 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| cospro | 0.80 | 0.10 | 0.02 | 0.90 | 495 | 215 | 15 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| cospro | 0.80 | 0.30 | 0.02 | 0.00 | 502 | 237 | 3 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Cormorant, Double crested Cormorant |
| cospro | 0.80 | 0.30 | 0.02 | 0.70 | 502 | 107 | 14 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| cospro | 0.80 | 0.30 | 0.02 | 0.90 | 502 | 217 | 11 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| meaning | 0.75 | 0.00 | 0.02 | 0.70 | 1257 | 112 | 51 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.75 | 0.30 | 0.02 | 0.70 | 1261 | 109 | 50 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.75 | 0.50 | 0.02 | 0.70 | 1281 | 105 | 45 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Vulnerable Native Breeds, Alsophila pometaria, Aristotelia chilensis, Ascalapha odorata, Atalopedes campestris, Blissus leucopterus, Bolboschoenus robustus, Calochilus paludosus |
| meaning | 0.80 | 0.00 | 0.02 | 0.00 | 1481 | 239 | 87 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.00 | 0.02 | 0.90 | 1481 | 220 | 75 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.30 | 0.02 | 0.00 | 1482 | 239 | 87 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.30 | 0.02 | 0.90 | 1482 | 220 | 75 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.50 | 0.02 | 0.00 | 1482 | 239 | 87 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.80 | 0.50 | 0.02 | 0.90 | 1482 | 220 | 75 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Alsophila pometaria, Ascalapha odorata, Calochilus paludosus, Chrysobalanus icaco, Chrysopogon zizanioides, Coregonus lavaretus, Dockrillia striolata, Dorotheanthus bellidiformis |
| meaning | 0.85 | 0.00 | 0.02 | 0.00 | 1633 | 242 | 52 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.00 | 0.02 | 0.90 | 1633 | 222 | 47 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.85 | 0.30 | 0.02 | 0.00 | 1633 | 242 | 52 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.30 | 0.02 | 0.90 | 1633 | 222 | 47 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.85 | 0.50 | 0.02 | 0.00 | 1633 | 242 | 52 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | American herring gull, European herring gull, Great black-backed gull, Ring billed Gull, Western Gull |
| meaning | 0.85 | 0.50 | 0.02 | 0.90 | 1633 | 222 | 47 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Dockrillia striolata, Dorotheanthus bellidiformis, Chrysopogon zizanioides, Guizotia abyssinica, Calochilus paludosus, Pholisora catullus, Lophocampa maculata, Microseris lanceolata |
| meaning | 0.90 | 0.00 | 0.02 | 0.00 | 1736 | 236 | 15 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.00 | 0.02 | 0.70 | 1736 | 107 | 18 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| meaning | 0.90 | 0.00 | 0.02 | 0.90 | 1736 | 216 | 19 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| meaning | 0.90 | 0.30 | 0.02 | 0.00 | 1736 | 236 | 15 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.30 | 0.02 | 0.70 | 1736 | 107 | 18 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| meaning | 0.90 | 0.30 | 0.02 | 0.90 | 1736 | 216 | 19 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
| meaning | 0.90 | 0.50 | 0.02 | 0.00 | 1736 | 236 | 15 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Great blue heron, Great heron, Grey heron, Heron |
| meaning | 0.90 | 0.50 | 0.02 | 0.70 | 1736 | 107 | 18 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Bamboo, Perching bird, Bird feeder, Nest box, Acridotheres, Finch, Bird food, Nightingale |
| meaning | 0.90 | 0.50 | 0.02 | 0.90 | 1736 | 216 | 19 | 0/8 | 0.00 | 0.060 | 3 | Pomacentridae | Pseudemys concinna concinna, Dorotheanthus bellidiformis, Pholisora catullus, Phasianidae, Moronidae, Lamnidae, Jacobaea vulgaris, Microseris lanceolata |
