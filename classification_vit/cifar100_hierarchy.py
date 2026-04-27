"""CIFAR-100 fine -> superclass mapping (standard torchvision ordering).

The 100 fine classes map onto 20 coarse superclasses. The mapping below is
the canonical CIFAR-100 grouping (Krizhevsky 2009 / torchvision).
"""

NUM_FINE = 100
NUM_SUPER = 20

# Coarse superclass ordering (torchvision):
#   0 aquatic_mammals    1 fish              2 flowers           3 food_containers
#   4 fruit_and_vegetables  5 household_electrical_devices
#   6 household_furniture   7 insects            8 large_carnivores
#   9 large_man-made_outdoor_things             10 large_natural_outdoor_scenes
#  11 large_omnivores_and_herbivores           12 medium-sized_mammals
#  13 non-insect_invertebrates                 14 people
#  15 reptiles            16 small_mammals    17 trees
#  18 vehicles_1          19 vehicles_2

FINE_TO_SUPER = {
    0:  4,  # apple              -> fruit_and_vegetables
    1:  1,  # aquarium_fish      -> fish
    2: 14,  # baby               -> people
    3:  8,  # bear               -> large_carnivores
    4:  0,  # beaver             -> aquatic_mammals
    5:  6,  # bed                -> household_furniture
    6:  7,  # bee                -> insects
    7:  7,  # beetle             -> insects
    8: 18,  # bicycle            -> vehicles_1
    9:  3,  # bottle             -> food_containers
    10: 3,  # bowl               -> food_containers
    11:14,  # boy                -> people
    12: 9,  # bridge             -> large_man-made_outdoor_things
    13:18,  # bus                -> vehicles_1
    14: 7,  # butterfly          -> insects
    15:11,  # camel              -> large_omnivores_and_herbivores
    16: 3,  # can                -> food_containers
    17: 9,  # castle             -> large_man-made_outdoor_things
    18: 7,  # caterpillar        -> insects
    19:11,  # cattle             -> large_omnivores_and_herbivores
    20: 6,  # chair              -> household_furniture
    21:11,  # chimpanzee         -> large_omnivores_and_herbivores
    22: 5,  # clock              -> household_electrical_devices
    23:10,  # cloud              -> large_natural_outdoor_scenes
    24: 7,  # cockroach          -> insects
    25: 6,  # couch              -> household_furniture
    26:13,  # crab               -> non-insect_invertebrates
    27:15,  # crocodile          -> reptiles
    28: 3,  # cup                -> food_containers
    29:15,  # dinosaur           -> reptiles
    30: 0,  # dolphin            -> aquatic_mammals
    31:11,  # elephant           -> large_omnivores_and_herbivores
    32: 1,  # flatfish           -> fish
    33:10,  # forest             -> large_natural_outdoor_scenes
    34:12,  # fox                -> medium-sized_mammals
    35:14,  # girl               -> people
    36:16,  # hamster            -> small_mammals
    37: 9,  # house              -> large_man-made_outdoor_things
    38:11,  # kangaroo           -> large_omnivores_and_herbivores
    39: 5,  # keyboard           -> household_electrical_devices
    40: 5,  # lamp               -> household_electrical_devices
    41:19,  # lawn_mower         -> vehicles_2
    42: 8,  # leopard            -> large_carnivores
    43: 8,  # lion               -> large_carnivores
    44:15,  # lizard             -> reptiles
    45:13,  # lobster            -> non-insect_invertebrates
    46:14,  # man                -> people
    47:17,  # maple_tree         -> trees
    48:18,  # motorcycle         -> vehicles_1
    49:10,  # mountain           -> large_natural_outdoor_scenes
    50:16,  # mouse              -> small_mammals
    51: 4,  # mushroom           -> fruit_and_vegetables
    52:17,  # oak_tree           -> trees
    53: 4,  # orange             -> fruit_and_vegetables
    54: 2,  # orchid             -> flowers
    55: 0,  # otter              -> aquatic_mammals
    56:17,  # palm_tree          -> trees
    57: 4,  # pear               -> fruit_and_vegetables
    58:18,  # pickup_truck       -> vehicles_1
    59:17,  # pine_tree          -> trees
    60:10,  # plain              -> large_natural_outdoor_scenes
    61: 3,  # plate              -> food_containers
    62: 2,  # poppy              -> flowers
    63:12,  # porcupine          -> medium-sized_mammals
    64:12,  # possum             -> medium-sized_mammals
    65:16,  # rabbit             -> small_mammals
    66:12,  # raccoon            -> medium-sized_mammals
    67: 1,  # ray                -> fish
    68: 9,  # road               -> large_man-made_outdoor_things
    69:19,  # rocket             -> vehicles_2
    70: 2,  # rose               -> flowers
    71:10,  # sea                -> large_natural_outdoor_scenes
    72: 0,  # seal               -> aquatic_mammals
    73: 1,  # shark              -> fish
    74:16,  # shrew              -> small_mammals
    75:12,  # skunk              -> medium-sized_mammals
    76: 9,  # skyscraper         -> large_man-made_outdoor_things
    77:13,  # snail              -> non-insect_invertebrates
    78:15,  # snake              -> reptiles
    79:13,  # spider             -> non-insect_invertebrates
    80:16,  # squirrel           -> small_mammals
    81:19,  # streetcar          -> vehicles_2
    82: 2,  # sunflower          -> flowers
    83: 4,  # sweet_pepper       -> fruit_and_vegetables
    84: 6,  # table              -> household_furniture
    85:19,  # tank               -> vehicles_2
    86: 5,  # telephone          -> household_electrical_devices
    87: 5,  # television         -> household_electrical_devices
    88: 8,  # tiger              -> large_carnivores
    89:19,  # tractor            -> vehicles_2
    90:18,  # train              -> vehicles_1
    91: 1,  # trout              -> fish
    92: 2,  # tulip              -> flowers
    93:15,  # turtle             -> reptiles
    94: 6,  # wardrobe           -> household_furniture
    95: 0,  # whale              -> aquatic_mammals
    96:17,  # willow_tree        -> trees
    97: 8,  # wolf               -> large_carnivores
    98:14,  # woman              -> people
    99:13,  # worm               -> non-insect_invertebrates
}

assert len(FINE_TO_SUPER) == NUM_FINE
assert set(FINE_TO_SUPER.values()) == set(range(NUM_SUPER))
