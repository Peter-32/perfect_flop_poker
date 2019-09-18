import os
import sys
import ast
import math
import itertools
import functools
import subprocess
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict


# RFI
lj_rfi = "55+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,AJo+,KQo+"
hj_rfi = "22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,65s,ATo+,KJo+,QJo"
co_rfi = "22+,A2s+,K6s+,Q8s+,J8s+,T8s+,97s+,86s+,75s+,65s,54s,ATo+,KTo+,QTo+,JTo"
bn_rfi = "22+,A2s+,K4s+,Q6s+,J7s+,T7s+,96s+,85s+,75s+,64s+,53s+,43s,A4o+,K9o+,Q9o+,J9o+,T9o"
sb_rfi = "22+,A2s+,K3s+,Q5s+,J6s+,T6s+,96s+,85s+,75s+,64s+,53s+,43s,A2o+,K8o+,Q9o+,J9o+,T9o,98o"

# vs RFI
vsRFI_bb_vs_lj__raise = "TT+,AQs+,AKo"
vsRFI_bb_vs_hj__raise = "TT+,ATs+,AKo"
vsRFI_bb_vs_co__raise = "99+,A8s+,KJs+,QJs,JTs,T9s,98s,AQo+,KQo"
vsRFI_bb_vs_bn__raise = "88+,A6s+,KTs+,QTs+,JTs,T9s,98s,87s,76s,AJo+,KQo"
vsRFI_bb_vs_sb__raise = "44+,A8s+,A5s-A3s,K9s+,Q9s+,J9s,T8s+,97s+,86s+,75s+,64s+,54s,AJo+,KQo"

vsRFI_bb_vs_lj__call = "99-22,AJs-A2s,K6s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,54s,AQo-ATo,KTo+,QTo+,JTo"
vsRFI_bb_vs_hj__call = "99-22,A9s-A2s,K5s+,Q7s+,J7s+,T7s+,96s+,86s+,75s+,65s,54s,43s,AQo-ATo,KTo+,QTo+,JTo"
vsRFI_bb_vs_co__call = "88-22,A7s-A2s,KTs-K2s,QTs-Q5s,J9s-J6s,T8s-T6s,97s-96s,85s+,74s+,64s+,53s+,43s,AJo-A7o,A5o,KJo-K9o,Q9o+,J9o+,T9o"
vsRFI_bb_vs_bn__call = "77-22,A5s-A2s,K9s-K2s,Q9s-Q2s,J9s-J5s,T8s-T5s,97s-95s,86s-84s,75s-74s,63s+,53s+,42s+,32s,ATo-A2o,KJo-K7o,Q8o+,J8o+,T8o+,98o,87o"
vsRFI_bb_vs_sb__call = "33-22,A7s-A6s,A2s,K8s-K2s,Q8s-Q2s,J8s-J4s,T7s-T4s,96s-94s,85s-84s,74s-73s,63s-62s,53s-52s,42s+,32s,ATo-A2o,KJo-K5o,Q7o+,J7o+,T7o+,97o+,87o,76o,65o"

# vs RFI
vsRFI_hj_vs_lj__raise = "JJ+,AJs+,KQs,AKo"
vsRFI_co_vs_lj__raise = "JJ+,AJs+,KQs,AKo"
vsRFI_co_vs_hj__raise = "JJ+,AJs+,A5s-A4s,KQs,T9s,AQo+"
vsRFI_bn_vs_lj__raise = "JJ+,AKs,A5s-A2s,AKo"
vsRFI_bn_vs_hj__raise = "JJ+,AJs+,A8s,A5s-A2s,KQs,76s,65s,54s,AKo"
vsRFI_bn_vs_co__raise = "TT+,ATs+,A7s-A2s,KJs+,QJs,JTs,T9s,76s,65s,54s,AJo+,KQo"
vsRFI_sb_vs_lj__raise = "JJ+,ATs+,KQs,AKo"
vsRFI_sb_vs_hj__raise = "JJ+,ATs+,KJs+,QJs,JTs,AQo+"
vsRFI_sb_vs_co__raise = "JJ+,A9s+,A5s-A4s,KJs+,QJs,JTs,T9s,98s,AJo+,KQo"
vsRFI_sb_vs_bn__raise = "55+,A2s+,K9s+,Q9s+,J9s+,T8s+,98s,87s,76s,ATo+,KQo+"

vsRFI_hj_vs_lj__call = "TT-77,ATs,KJs,QJs,JTs,AQo"
vsRFI_co_vs_lj__call = "TT-66,ATs,KJs,QJs,JTs,AQo"
vsRFI_co_vs_hj__call = "TT-55,ATs,KJs,QJs,JTs"
vsRFI_bn_vs_lj__call = "TT-55,AQs-ATs,KTs+,QTs+,JTs,T9s,98s,87s,AQo"
vsRFI_bn_vs_hj__call = "TT-33,ATs-A9s,KJs-KTs,QTs+,JTs,T9s,98s,87s,AQo-AJo,KQo"
vsRFI_bn_vs_co__call = "99-22,A9s-A8s,KTs,QTs,98s,87s"
vsRFI_sb_vs_lj__call = "TT-77,QJs,JTs,AQo"
vsRFI_sb_vs_hj__call = "TT-66,T9s"
vsRFI_sb_vs_co__call = "TT-55,KTs,QTs"
vsRFI_sb_vs_bn__call = ""

# These look similar
RFIvs3B_lj_vs_hjco_call = "JJ-77,AQs-AJs,KQs,QJs,JTs"
RFIvs3B_lj_vs_bn_call = "JJ-77,AQs-AJs,KJs+,QJs,JTs,T9s"
RFIvs3B_lj_vs_blinds_call = "JJ-66,AQs-ATs,KJs+,QJs,JTs,T9s"
RFIvs3B_hj_vs_ahead_call = "TT-66,AQs-ATs,KTs+,QTs+,JTs,T9s,AQo"
RFIvs3B_co_vs_bn_call = "TT-66,AQs-A8s,KTs+,QTs+,JTs,T9s,98s,AQo"
RFIvs3B_co_vs_blinds_call = "TT-55,AQs-A8s,KTs+,QTs+,JTs,T9s,98s,87s,AQo"
RFIvs3B_bnsb_vs_ahead_call = "TT-33,AQs-A6s,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,65s,54s,ATo+,KJo+,QJo"


# Input: range
m = {
    "A": 14, "K": 13, "Q": 12, "J": 11, "T": 10, "9": 9,
    "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2
}
def range_to_hands(c_range="JJ+,AJs+,KQs,AKo"):

    temp = c_range.split(",")

    pps = []
    pp = temp[0]
    if "+" in pp:
        for i in range(14,m[pp[0]]-1,-1):
            pps.append([i, i])
    elif "-" in pp:
        for i in range(m[pp[0]],m[pp[-1]]-1,-1):
            pps.append([i, i])
    else:
        pps.append([m[pp[0]], m[pp[0]]])

    ss = []
    temp_s = [x for x in temp if "s" in x]
    for s in temp_s:
        if "+" in s:
            for i in range(m[s[0]]-1,m[s[1]]-1,-1):
                ss.append([m[s[0]], i])
        elif "-" in s:
            for i in range(m[s[1]],m[s[5]]-1,-1):
                ss.append([m[s[0]], i])
        else:
            ss.append([m[s[0]], m[s[1]]])

    os = []
    temp_o = [x for x in temp if "o" in x]
    for o in temp_o:
        if "+" in o:
            for i in range(m[o[0]]-1,m[o[1]]-1,-1):
                os.append([m[o[0]], i])
        elif "-" in o:
            for i in range(m[o[1]],m[o[5]]-1,-1):
                os.append([m[o[0]], i])
        else:
            os.append([m[o[0]], m[o[1]]])
    # Output: [[2,2]], [[14,13]], [[14,13]]
        # PP, Suited, Offsuit
    return pps, ss, os


cat1_rankings = ["set", "trips", "two pair", "overpair 9+", "any overpair", "TP J-kicker",
                 "TP K-kicker", "TP any kicker"]
cat2_nonpaired_rankings = ["top pair bad kicker", "middle pair", "bottom pair", "PP below middle pair",
                           "AJ high", "KQ high", "KJ high bdfd", "K8 high bdfd", ]
cat2_paired_rankings = ["Ace high", "PP below top card", "KQ high", "all"]
cat3_rankings = ["FD", "OESD", "Gutshot", "3 to a straight not all from low end",
                 "3 to a straight low end bdfd", "3 to a straight low end",
                 "5 cards within 7 values with bdfd", "Q- high bdfd",
                 "3 cards within 4 values as overcards", "A- bdfd"]
first_cat4_pp_rankings = ["JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22"]



def my_hands_cat1_level_x_and_above(x):
    result = [[], [], []]
    if x >= 1:
        result[1] += my_hands_s_straight
        result[2] += my_hands_o_straight
    if x >= 2:
        result[0] += my_hands_pp_sets
    if x >= 3:
        result[1] += my_hands_s_trips
        result[2] += my_hands_o_trips
    if x >= 4:
        result[1] += my_hands_s_two_pair
        result[2] += my_hands_o_two_pair
    if x >= 5:
        result[0] += my_hands_pp_overpair_9plus
    if x >= 6:
        result[0] += my_hands_pp_any_overpair
    if x >= 7:
        result[1] += my_hands_s_tp_k_kicker
        result[2] += my_hands_o_tp_k_kicker
    if x >= 8:
        result[1] += my_hands_s_tp_j_kicker
        result[2] += my_hands_o_tp_j_kicker
    if x >= 9:
        result[1] += my_hands_s_tp_any_kicker
        result[2] += my_hands_o_tp_any_kicker

    result[0].sort(reverse=True)
    result[1].sort(reverse=True)
    result[2].sort(reverse=True)
    result[0] = list(k for k,_ in itertools.groupby(result[0]))
    result[1] = list(k for k,_ in itertools.groupby(result[1]))
    result[2] = list(k for k,_ in itertools.groupby(result[2]))

    # Return result
    my_hands_cat1 = result
    return my_hands_cat1

# Performance improvement by filtering out cat1 from hands already, but would also need a copy of hands
def my_hands_cat2_level_x_and_above(x, my_hands_cat1):
    result = [[], [], []]
    if x >= 1:
        # Cat 1
        result[1] += my_hands_s_straight
        result[2] += my_hands_o_straight
        result[0] += my_hands_pp_sets
        result[1] += my_hands_s_trips
        result[2] += my_hands_o_trips
        result[1] += my_hands_s_two_pair
        result[2] += my_hands_o_two_pair
        result[0] += my_hands_pp_overpair_9plus
        result[0] += my_hands_pp_any_overpair
        result[1] += my_hands_s_tp_k_kicker
        result[2] += my_hands_o_tp_k_kicker
        result[1] += my_hands_s_tp_j_kicker
        result[2] += my_hands_o_tp_j_kicker
        result[1] += my_hands_s_tp_any_kicker
        result[2] += my_hands_o_tp_any_kicker

        # Cat 2
        result[1] += my_hands_s_tp_bad_kicker
        result[2] += my_hands_o_tp_bad_kicker
    if x >= 2:
        result[1] += my_hands_s_middle_pair
        result[2] += my_hands_o_middle_pair
    if x >= 3:
        result[0] += my_hands_pp_below_top_pair
    if x >= 4:
        result[1] += my_hands_s_bottom_pair
        result[2] += my_hands_o_bottom_pair
    if x >= 5:
        result[1] += my_hands_s_aj_high
        result[2] += my_hands_o_aj_high
    if x >= 6:
        result[0] += my_hands_pp_below_middle_pair
    if x >= 7:
        result[1] += my_hands_s_kq_high
        result[2] += my_hands_o_kq_high
    if x >= 8:
        result[0] += my_hands_pp_below_bottom_pair
    if x >= 9:
        result[1] += my_hands_s_kj_high
        result[2] += my_hands_o_kj_high
    if x >= 10:
        result[1] += my_hands_s_k8_high
        result[2] += my_hands_o_k8_high

    result[0].sort(reverse=True)
    result[1].sort(reverse=True)
    result[2].sort(reverse=True)
    result[0] = list(k for k,_ in itertools.groupby(result[0]))
    result[1] = list(k for k,_ in itertools.groupby(result[1]))
    result[2] = list(k for k,_ in itertools.groupby(result[2]))

    # Interim
    cat1_unique_pp = [x for (x,y) in my_hands_cat1[0]]
    cat1_unique_s = [x for (x,y) in my_hands_cat1[1]]
    cat1_unique_o = [x for (x,y) in my_hands_cat1[2]]

    # Remove cat1 from these cat2s
    result[0] = [(x,y) for (x,y) in result[0] if x not in cat1_unique_pp]
    result[1] = [(x,y) for (x,y) in result[1] if x not in cat1_unique_s]
    result[2] = [(x,y) for (x,y) in result[2] if x not in cat1_unique_o]

    # Return result
    my_hands_cat2 = result
    return my_hands_cat2

# Performance improvement by filtering out cat1+cat2 from hands already, but would also need a copy of hands
def my_hands_cat3_level_x_and_above(x, my_hands_cat1, my_hands_cat2):
    bdfd_result = [[], [], []]
    other_result = [[], [], []]
    result = [[], [], []]
    if x >= 1:
        other_result[0] += my_hands_pp_fd
        other_result[1] += my_hands_s_fd
        other_result[2] += my_hands_o_fd
    if x >= 2:
        other_result[0] += my_hands_pp_oesd
        other_result[1] += my_hands_s_oesd
        other_result[2] += my_hands_o_oesd
    if x >= 3:
        other_result[0] += my_hands_pp_gutshot
        other_result[1] += my_hands_s_gutshot
        other_result[2] += my_hands_o_gutshot
    if x >= 4:
        other_result[1] += my_hands_s_3_to_straight_not_all_from_low_end
        other_result[2] += my_hands_o_3_to_straight_not_all_from_low_end
    if x >= 5:
        bdfd_result[1] += my_hands_s_3_to_straight_low_end_bdfd
        bdfd_result[2] += my_hands_o_3_to_straight_low_end_bdfd
    if x >= 6:
        other_result[1] += my_hands_s_3_to_straight_low_end
        other_result[2] += my_hands_o_3_to_straight_low_end
    if x >= 7:
        bdfd_result[1] += my_hands_s_5_unique_cards_within_7_values_bdfd
        bdfd_result[2] += my_hands_o_5_unique_cards_within_7_values_bdfd
    if x >= 8:
        bdfd_result[0] += my_hands_pp_q_minus_bdfd
        bdfd_result[1] += my_hands_s_q_minus_bdfd
        bdfd_result[2] += my_hands_o_q_minus_bdfd
    if x >= 9:
        other_result[1] += my_hands_s_lowest_card_is_one_of_3_cards_within_4_values_and_two_overcards
        other_result[2] += my_hands_o_lowest_card_is_one_of_3_cards_within_4_values_and_two_overcards
    if x >= 10:
        bdfd_result[0] += my_hands_pp_a_minus_bdfd
        bdfd_result[1] += my_hands_s_a_minus_bdfd
        bdfd_result[2] += my_hands_o_a_minus_bdfd

    # Remove duplicates within bdfd hands
    bdfd_result[0].sort(reverse=True)
    bdfd_result[1].sort(reverse=True)
    bdfd_result[2].sort(reverse=True)
    bdfd_result[0] = list(k for k,_ in itertools.groupby(bdfd_result[0]))
    bdfd_result[1] = list(k for k,_ in itertools.groupby(bdfd_result[1]))
    bdfd_result[2] = list(k for k,_ in itertools.groupby(bdfd_result[2]))

    # Add all together
    result[0] = bdfd_result[0] + other_result[0]
    result[1] = bdfd_result[1] + other_result[1]
    result[2] = bdfd_result[2] + other_result[2]

    # Reduce with max combos number used and sort
    groupby_dict = defaultdict(int)
    for val in result[0]:
        groupby_dict[tuple(val[0])] += val[1]
    result[0] = [(sorted(list(x), reverse=True),min(y, 6)) for (x,y) in groupby_dict.items()]

    groupby_dict = defaultdict(int)
    for val in result[1]:
        groupby_dict[tuple(val[0])] += val[1]
    result[1] = [(sorted(list(x), reverse=True),min(y, 4)) for (x,y) in groupby_dict.items()]

    groupby_dict = defaultdict(int)
    for val in result[2]:
        groupby_dict[tuple(val[0])] += val[1]
    result[2] = [(sorted(list(x), reverse=True),min(y, 12)) for (x,y) in groupby_dict.items()]

    # Interim
    cat1_unique_pp = [x for (x,y) in my_hands_cat1[0]]
    cat1_unique_s = [x for (x,y) in my_hands_cat1[1]]
    cat1_unique_o = [x for (x,y) in my_hands_cat1[2]]
    cat2_unique_pp = [x for (x,y) in my_hands_cat2[0]]
    cat2_unique_s = [x for (x,y) in my_hands_cat2[1]]
    cat2_unique_o = [x for (x,y) in my_hands_cat2[2]]

    # Remove cat1 and cat2
    result[0] = [(x,y) for (x,y) in result[0] if x not in cat1_unique_pp and x not in cat2_unique_pp]
    result[1] = [(x,y) for (x,y) in result[1] if x not in cat1_unique_s and x not in cat2_unique_s]
    result[2] = [(x,y) for (x,y) in result[2] if x not in cat1_unique_o and x not in cat2_unique_o]

    # Add cat2 hands
    if x >= 11:
        result[1] += [(x,y) for (x,y) in my_hands_s_k8_high if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in my_hands_o_k8_high if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 12:
        result[1] += [(x,y) for (x,y) in my_hands_s_kj_high if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in my_hands_o_kj_high if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 13:
        result[0] += [(x,y) for (x,y) in my_hands_pp_below_bottom_pair if x not in cat1_unique_pp and x not in cat2_unique_pp]
    if x >= 14:
        result[1] += [(x,y) for (x,y) in my_hands_s_kq_high if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in my_hands_o_kq_high if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 15:
        result[0] += [(x,y) for (x,y) in my_hands_pp_below_middle_pair if x not in cat1_unique_pp and x not in cat2_unique_pp]
    # Add cat4 hands
    if x >= 16:
        remaining_cat2_type_hands_pp = [x for (x,y) in my_hands_pp_below_top_pair]
        remaining_cat2_type_hands_s = [x for (x,y) in my_hands_s_aj_high] + [x for (x,y) in my_hands_s_bottom_pair] + [x for (x,y) in my_hands_s_middle_pair] + [x for (x,y) in my_hands_s_tp_bad_kicker]
        remaining_cat2_type_hands_o = [x for (x,y) in my_hands_o_aj_high] + [x for (x,y) in my_hands_o_bottom_pair] + [x for (x,y) in my_hands_o_middle_pair] + [x for (x,y) in my_hands_o_tp_bad_kicker]
        result[0] += [(x, 6) for x in my_hands[0] if x not in cat1_unique_pp and x not in cat2_unique_pp and x not in remaining_cat2_type_hands_pp]
        result[1] += [(x, 4) for x in my_hands[1] if x not in cat1_unique_s and x not in cat2_unique_s and x not in remaining_cat2_type_hands_s]
        result[2] += [(x, 12) for x in my_hands[2] if x not in cat1_unique_o and x not in cat2_unique_o and x not in remaining_cat2_type_hands_o]
    # Add cat2 hands with pairs
    if x >= 17:
        result[1] += [(x,y) for (x,y) in my_hands_s_aj_high if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in my_hands_o_aj_high if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 18:
        result[1] += [(x,y) for (x,y) in my_hands_s_bottom_pair if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in my_hands_o_bottom_pair if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 19:
        result[0] += [(x,y) for (x,y) in my_hands_pp_below_top_pair if x not in cat1_unique_pp and x not in cat2_unique_pp]
    if x >= 20:
        result[1] += [(x,y) for (x,y) in my_hands_s_middle_pair if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in my_hands_o_middle_pair if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 21:
        result[1] += [(x,y) for (x,y) in my_hands_s_tp_bad_kicker if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in my_hands_o_tp_bad_kicker if x not in cat1_unique_o and x not in cat2_unique_o]

    # Reduce with max combos number used and sort
    groupby_dict = defaultdict(int)
    for val in result[0]:
        groupby_dict[tuple(val[0])] = max(groupby_dict[tuple(val[0])], val[1])
    result[0] = [(sorted(list(x), reverse=True),min(y, 6)) for (x,y) in groupby_dict.items()]

    groupby_dict = defaultdict(int)
    for val in result[1]:
        groupby_dict[tuple(val[0])] = max(groupby_dict[tuple(val[0])], val[1])
    result[1] = [(sorted(list(x), reverse=True),min(y, 4)) for (x,y) in groupby_dict.items()]

    groupby_dict = defaultdict(int)
    for val in result[2]:
        groupby_dict[tuple(val[0])] = max(groupby_dict[tuple(val[0])], val[1])
    result[2] = [(sorted(list(x), reverse=True),min(y, 12)) for (x,y) in groupby_dict.items()]

    # Return results
    my_hands_cat3 = result
    return my_hands_cat3


def opponents_hands_cat1_level_x_and_above(x):
    result = [[], [], []]
    if x >= 1:
        result[1] += opponents_hands_s_straight
        result[2] += opponents_hands_o_straight
    if x >= 2:
        result[0] += opponents_hands_pp_sets
    if x >= 3:
        result[1] += opponents_hands_s_trips
        result[2] += opponents_hands_o_trips
    if x >= 4:
        result[1] += opponents_hands_s_two_pair
        result[2] += opponents_hands_o_two_pair
    if x >= 5:
        result[0] += opponents_hands_pp_overpair_9plus
    if x >= 6:
        result[0] += opponents_hands_pp_any_overpair
    if x >= 7:
        result[1] += opponents_hands_s_tp_k_kicker
        result[2] += opponents_hands_o_tp_k_kicker
    if x >= 8:
        result[1] += opponents_hands_s_tp_j_kicker
        result[2] += opponents_hands_o_tp_j_kicker
    if x >= 9:
        result[1] += opponents_hands_s_tp_any_kicker
        result[2] += opponents_hands_o_tp_any_kicker

    result[0].sort(reverse=True)
    result[1].sort(reverse=True)
    result[2].sort(reverse=True)
    result[0] = list(k for k,_ in itertools.groupby(result[0]))
    result[1] = list(k for k,_ in itertools.groupby(result[1]))
    result[2] = list(k for k,_ in itertools.groupby(result[2]))

    # Return result
    opponents_hands_cat1 = result
    return opponents_hands_cat1

# Performance improvement by filtering out cat1 from hands already, but would also need a copy of hands
def opponents_hands_cat2_level_x_and_above(x, opponents_hands_cat1):
    result = [[], [], []]
    if x >= 1:
        # Cat 1
        result[1] += opponents_hands_s_straight
        result[2] += opponents_hands_o_straight
        result[0] += opponents_hands_pp_sets
        result[1] += opponents_hands_s_trips
        result[2] += opponents_hands_o_trips
        result[1] += opponents_hands_s_two_pair
        result[2] += opponents_hands_o_two_pair
        result[0] += opponents_hands_pp_overpair_9plus
        result[0] += opponents_hands_pp_any_overpair
        result[1] += opponents_hands_s_tp_k_kicker
        result[2] += opponents_hands_o_tp_k_kicker
        result[1] += opponents_hands_s_tp_j_kicker
        result[2] += opponents_hands_o_tp_j_kicker
        result[1] += opponents_hands_s_tp_any_kicker
        result[2] += opponents_hands_o_tp_any_kicker

        # Cat 2
        result[1] += opponents_hands_s_tp_bad_kicker
        result[2] += opponents_hands_o_tp_bad_kicker
    if x >= 2:
        result[1] += opponents_hands_s_middle_pair
        result[2] += opponents_hands_o_middle_pair
    if x >= 3:
        result[0] += opponents_hands_pp_below_top_pair
    if x >= 4:
        result[1] += opponents_hands_s_bottom_pair
        result[2] += opponents_hands_o_bottom_pair
    if x >= 5:
        result[1] += opponents_hands_s_aj_high
        result[2] += opponents_hands_o_aj_high
    if x >= 6:
        result[0] += opponents_hands_pp_below_middle_pair
    if x >= 7:
        result[1] += opponents_hands_s_kq_high
        result[2] += opponents_hands_o_kq_high
    if x >= 8:
        result[0] += opponents_hands_pp_below_bottom_pair
    if x >= 9:
        result[1] += opponents_hands_s_kj_high
        result[2] += opponents_hands_o_kj_high
    if x >= 10:
        result[1] += opponents_hands_s_k8_high
        result[2] += opponents_hands_o_k8_high

    result[0].sort(reverse=True)
    result[1].sort(reverse=True)
    result[2].sort(reverse=True)
    result[0] = list(k for k,_ in itertools.groupby(result[0]))
    result[1] = list(k for k,_ in itertools.groupby(result[1]))
    result[2] = list(k for k,_ in itertools.groupby(result[2]))

    # Interim
    cat1_unique_pp = [x for (x,y) in opponents_hands_cat1[0]]
    cat1_unique_s = [x for (x,y) in opponents_hands_cat1[1]]
    cat1_unique_o = [x for (x,y) in opponents_hands_cat1[2]]

    # Remove cat1 from these cat2s
    result[0] = [(x,y) for (x,y) in result[0] if x not in cat1_unique_pp]
    result[1] = [(x,y) for (x,y) in result[1] if x not in cat1_unique_s]
    result[2] = [(x,y) for (x,y) in result[2] if x not in cat1_unique_o]

    # Return result
    opponents_hands_cat2 = result
    return opponents_hands_cat2

# Performance improvement by filtering out cat1+cat2 from hands already, but would also need a copy of hands
def opponents_hands_cat3_level_x_and_above(x, opponents_hands_cat1, opponents_hands_cat2, skip_4_to_10_and_13_to_15=True):
    bdfd_result = [[], [], []]
    other_result = [[], [], []]
    result = [[], [], []]
    if x >= 1:
        other_result[0] += opponents_hands_pp_fd
        other_result[1] += opponents_hands_s_fd
        other_result[2] += opponents_hands_o_fd
    if x >= 2:
        other_result[0] += opponents_hands_pp_oesd
        other_result[1] += opponents_hands_s_oesd
        other_result[2] += opponents_hands_o_oesd
    if x >= 3:
        other_result[0] += opponents_hands_pp_gutshot
        other_result[1] += opponents_hands_s_gutshot
        other_result[2] += opponents_hands_o_gutshot
    if x >= 4 and not skip_4_to_10_and_13_to_15:
        other_result[1] += opponents_hands_s_3_to_straight_not_all_from_low_end
        other_result[2] += opponents_hands_o_3_to_straight_not_all_from_low_end
    if x >= 5 and not skip_4_to_10_and_13_to_15:
        bdfd_result[1] += opponents_hands_s_3_to_straight_low_end_bdfd
        bdfd_result[2] += opponents_hands_o_3_to_straight_low_end_bdfd
    if x >= 6 and not skip_4_to_10_and_13_to_15:
        other_result[1] += opponents_hands_s_3_to_straight_low_end
        other_result[2] += opponents_hands_o_3_to_straight_low_end
    if x >= 7 and not skip_4_to_10_and_13_to_15:
        bdfd_result[1] += opponents_hands_s_5_unique_cards_within_7_values_bdfd
        bdfd_result[2] += opponents_hands_o_5_unique_cards_within_7_values_bdfd
    if x >= 8 and not skip_4_to_10_and_13_to_15:
        bdfd_result[0] += opponents_hands_pp_q_minus_bdfd
        bdfd_result[1] += opponents_hands_s_q_minus_bdfd
        bdfd_result[2] += opponents_hands_o_q_minus_bdfd
    if x >= 9:
        other_result[1] += opponents_hands_s_lowest_card_is_one_of_3_cards_within_4_values_and_two_overcards
        other_result[2] += opponents_hands_o_lowest_card_is_one_of_3_cards_within_4_values_and_two_overcards
    if x >= 10 and not skip_4_to_10_and_13_to_15:
        bdfd_result[0] += opponents_hands_pp_a_minus_bdfd
        bdfd_result[1] += opponents_hands_s_a_minus_bdfd
        bdfd_result[2] += opponents_hands_o_a_minus_bdfd

    # Remove duplicates within bdfd hands
    bdfd_result[0].sort(reverse=True)
    bdfd_result[1].sort(reverse=True)
    bdfd_result[2].sort(reverse=True)
    bdfd_result[0] = list(k for k,_ in itertools.groupby(bdfd_result[0]))
    bdfd_result[1] = list(k for k,_ in itertools.groupby(bdfd_result[1]))
    bdfd_result[2] = list(k for k,_ in itertools.groupby(bdfd_result[2]))

    # Add all together
    result[0] = bdfd_result[0] + other_result[0]
    result[1] = bdfd_result[1] + other_result[1]
    result[2] = bdfd_result[2] + other_result[2]

    # Reduce with max combos number used and sort
    groupby_dict = defaultdict(int)
    for val in result[0]:
        groupby_dict[tuple(val[0])] += val[1]
    result[0] = [(sorted(list(x), reverse=True),min(y, 6)) for (x,y) in groupby_dict.items()]

    groupby_dict = defaultdict(int)
    for val in result[1]:
        groupby_dict[tuple(val[0])] += val[1]
    result[1] = [(sorted(list(x), reverse=True),min(y, 4)) for (x,y) in groupby_dict.items()]

    groupby_dict = defaultdict(int)
    for val in result[2]:
        groupby_dict[tuple(val[0])] += val[1]
    result[2] = [(sorted(list(x), reverse=True),min(y, 12)) for (x,y) in groupby_dict.items()]

    # Interim
    cat1_unique_pp = [x for (x,y) in opponents_hands_cat1[0]]
    cat1_unique_s = [x for (x,y) in opponents_hands_cat1[1]]
    cat1_unique_o = [x for (x,y) in opponents_hands_cat1[2]]
    cat2_unique_pp = [x for (x,y) in opponents_hands_cat2[0]]
    cat2_unique_s = [x for (x,y) in opponents_hands_cat2[1]]
    cat2_unique_o = [x for (x,y) in opponents_hands_cat2[2]]

    # Remove cat1 and cat2
    result[0] = [(x,y) for (x,y) in result[0] if x not in cat1_unique_pp and x not in cat2_unique_pp]
    result[1] = [(x,y) for (x,y) in result[1] if x not in cat1_unique_s and x not in cat2_unique_s]
    result[2] = [(x,y) for (x,y) in result[2] if x not in cat1_unique_o and x not in cat2_unique_o]

    # Add cat2 hands
    if x >= 11 and not skip_4_to_10_and_13_to_15:
        result[1] += [(x,y) for (x,y) in opponents_hands_s_k8_high if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in opponents_hands_o_k8_high if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 12 and not skip_4_to_10_and_13_to_15:
        result[1] += [(x,y) for (x,y) in opponents_hands_s_kj_high if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in opponents_hands_o_kj_high if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 13 and not skip_4_to_10_and_13_to_15:
        result[0] += [(x,y) for (x,y) in opponents_hands_pp_below_bottom_pair if x not in cat1_unique_pp and x not in cat2_unique_pp]
    if x >= 14 and not skip_4_to_10_and_13_to_15:
        result[1] += [(x,y) for (x,y) in opponents_hands_s_kq_high if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in opponents_hands_o_kq_high if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 15 and not skip_4_to_10_and_13_to_15:
        result[0] += [(x,y) for (x,y) in opponents_hands_pp_below_middle_pair if x not in cat1_unique_pp and x not in cat2_unique_pp]
    # Add cat4 hands
    if x >= 16:
        remaining_cat2_type_hands_pp = [x for (x,y) in opponents_hands_pp_below_bottom_pair] + [x for (x,y) in opponents_hands_pp_below_middle_pair] + [x for (x,y) in opponents_hands_pp_below_top_pair]
        remaining_cat2_type_hands_s = [x for (x,y) in opponents_hands_s_k8_high] + [x for (x,y) in opponents_hands_s_kj_high] + [x for (x,y) in opponents_hands_s_kq_high] + [x for (x,y) in opponents_hands_s_aj_high] + [x for (x,y) in opponents_hands_s_bottom_pair] + [x for (x,y) in opponents_hands_s_middle_pair] + [x for (x,y) in opponents_hands_s_tp_bad_kicker]
        remaining_cat2_type_hands_o = [x for (x,y) in opponents_hands_o_k8_high] + [x for (x,y) in opponents_hands_o_kj_high] + [x for (x,y) in opponents_hands_o_kq_high] + [x for (x,y) in opponents_hands_o_aj_high] + [x for (x,y) in opponents_hands_o_bottom_pair] + [x for (x,y) in opponents_hands_o_middle_pair] + [x for (x,y) in opponents_hands_o_tp_bad_kicker]
        result[0] += [(x, 6) for x in opponents_hands[0] if x not in cat1_unique_pp and x not in cat2_unique_pp and x not in remaining_cat2_type_hands_pp]
        result[1] += [(x, 4) for x in opponents_hands[1] if x not in cat1_unique_s and x not in cat2_unique_s and x not in remaining_cat2_type_hands_s]
        result[2] += [(x, 12) for x in opponents_hands[2] if x not in cat1_unique_o and x not in cat2_unique_o and x not in remaining_cat2_type_hands_o]
    # Add cat2 hands with pairs
    if x >= 17:
        result[1] += [(x,y) for (x,y) in opponents_hands_s_aj_high if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in opponents_hands_o_aj_high if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 18:
        result[1] += [(x,y) for (x,y) in opponents_hands_s_bottom_pair if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in opponents_hands_o_bottom_pair if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 19:
        result[0] += [(x,y) for (x,y) in opponents_hands_pp_below_top_pair if x not in cat1_unique_pp and x not in cat2_unique_pp]
    if x >= 20:
        result[1] += [(x,y) for (x,y) in opponents_hands_s_middle_pair if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in opponents_hands_o_middle_pair if x not in cat1_unique_o and x not in cat2_unique_o]
    if x >= 21:
        result[1] += [(x,y) for (x,y) in opponents_hands_s_tp_bad_kicker if x not in cat1_unique_s and x not in cat2_unique_s]
        result[2] += [(x,y) for (x,y) in opponents_hands_o_tp_bad_kicker if x not in cat1_unique_o and x not in cat2_unique_o]

    # Reduce with max combos number used and sort
    groupby_dict = defaultdict(int)
    for val in result[0]:
        groupby_dict[tuple(val[0])] = max(groupby_dict[tuple(val[0])], val[1])
    result[0] = [(sorted(list(x), reverse=True),min(y, 6)) for (x,y) in groupby_dict.items()]

    groupby_dict = defaultdict(int)
    for val in result[1]:
        groupby_dict[tuple(val[0])] = max(groupby_dict[tuple(val[0])], val[1])
    result[1] = [(sorted(list(x), reverse=True),min(y, 4)) for (x,y) in groupby_dict.items()]

    groupby_dict = defaultdict(int)
    for val in result[2]:
        groupby_dict[tuple(val[0])] = max(groupby_dict[tuple(val[0])], val[1])
    result[2] = [(sorted(list(x), reverse=True),min(y, 12)) for (x,y) in groupby_dict.items()]

    # Return results
    opponents_hands_cat3 = result
    return opponents_hands_cat3





opponent_unraised_strategy = None # To be defined later; changes by flop
opponent_raised_strategy = {
    'cat1': {1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 6},
    'cat2': {1: 3, 2: 5, 3: 6, 4: 6, 5: 7, 6: 7, 7: 7},
    'cat3': {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 5, 7: 5},
}
opponent_reraised_strategy = {
    'cat1': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
    'cat2': {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3},
    'cat3': {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3},
}
opponent_strategy = None
def get_flop_type_number():
    return \
    7 if flop[0] == flop[1] or flop[1] == flop[2] else \
    1 if flop[0] >= 13 and flop[1] >= 13 else \
    2 if flop[0] >= 13 and flop[1] >= 9 else \
    3 if flop[0] >= 13 else \
    4 if flop[0] >= 10 and flop[1] >= 9 else \
    5 if flop[0] >= 10 else \
    6
def get_opponent_situation(bets):
    return \
    "oop_open" if bets == 0 and my_position_ip == True else \
    "oop_vs_cb" if bets == 1 and my_position_ip == True else \
    "oop_vs_br" if bets >= 2 and my_position_ip == True else \
    "ip_vs_c" if bets == 0 and my_position_ip == False else \
    "ip_vs_b" if bets == 1 and my_position_ip == False else \
    "ip_vs_cbr"


#####

flops = []
for rank1 in range(14,1,-1):
    for rank2 in range(14,1,-1):
        for rank3 in range(14,1,-1):
            if rank1 >= rank2 and rank2 >= rank3:
                flops.append([rank1, rank2, rank3])

# # Start from a spot
# flops = flops[406:]


range_names = ["00 LJ vs BB 3Bet", "01 LJ vs BB Call", "02 LJ vs HJ 3Bet", "03 LJ vs CO 3Bet", "04 LJ vs BN 3Bet", "05 LJ vs SB 3Bet", "06 LJ vs HJ Call", "07 LJ vs CO Call", "08 LJ vs BN Call", "09 LJ vs SB Call", "10 HJ vs BB 3Bet", "11 HJ vs BB Call", "12 HJ vs CO 3Bet", "13 HJ vs BN 3Bet", "14 HJ vs SB 3Bet", "15 HJ vs CO Call", "16 HJ vs BN Call", "17 HJ vs SB Call", "18 CO vs BB 3Bet", "19 CO vs BB Call", "20 CO vs BN 3Bet", "21 CO vs SB 3Bet", "22 CO vs BN Call", "23 CO vs SB Call", "24 BN vs BB 3Bet", "25 BN vs BB Call", "26 BN vs SB 3Bet", "27 SB vs BB 3Bet", "28 SB vs BB Call", "29 BB 3Bet vs LJ", "30 BB Call vs LJ", "31 HJ 3Bet vs LJ", "32 CO 3Bet vs LJ", "33 BN 3Bet vs LJ", "34 SB 3Bet vs LJ", "35 HJ Call vs LJ", "36 CO Call vs LJ", "37 BN Call vs LJ", "38 SB Call vs LJ", "39 BB 3Bet vs HJ", "40 BB Call vs HJ", "41 CO 3Bet vs HJ", "42 BN 3Bet vs HJ", "43 SB 3Bet vs HJ", "44 CO Call vs HJ", "45 BN Call vs HJ", "46 SB Call vs HJ", "47 BB 3Bet vs CO", "48 BB Call vs CO", "49 BN 3Bet vs CO", "50 SB 3Bet vs CO", "51 BN Call vs CO", "52 SB Call vs CO", "53 BB 3Bet vs BN", "54 BB Call vs BN", "55 SB 3Bet vs BN", "56 BB 3Bet vs SB", "57 BB Call vs SB",]
my_ranges = [RFIvs3B_lj_vs_blinds_call, lj_rfi, RFIvs3B_lj_vs_hjco_call, RFIvs3B_lj_vs_hjco_call, RFIvs3B_lj_vs_bn_call, RFIvs3B_lj_vs_blinds_call, lj_rfi, lj_rfi, lj_rfi, lj_rfi, RFIvs3B_hj_vs_ahead_call, hj_rfi, RFIvs3B_hj_vs_ahead_call, RFIvs3B_hj_vs_ahead_call, RFIvs3B_hj_vs_ahead_call, hj_rfi, hj_rfi, hj_rfi, RFIvs3B_co_vs_blinds_call, co_rfi, RFIvs3B_co_vs_bn_call, RFIvs3B_co_vs_blinds_call, co_rfi, co_rfi, RFIvs3B_bnsb_vs_ahead_call, bn_rfi, RFIvs3B_bnsb_vs_ahead_call, RFIvs3B_bnsb_vs_ahead_call, sb_rfi, vsRFI_bb_vs_lj__raise, vsRFI_bb_vs_lj__call, vsRFI_hj_vs_lj__raise, vsRFI_co_vs_lj__raise, vsRFI_bn_vs_lj__raise, vsRFI_sb_vs_lj__raise, vsRFI_hj_vs_lj__call, vsRFI_co_vs_lj__call, vsRFI_bn_vs_lj__call, vsRFI_sb_vs_lj__call, vsRFI_bb_vs_hj__raise, vsRFI_bb_vs_hj__call, vsRFI_co_vs_hj__raise, vsRFI_bn_vs_hj__raise, vsRFI_sb_vs_hj__raise, vsRFI_co_vs_hj__call, vsRFI_bn_vs_hj__call, vsRFI_sb_vs_hj__call, vsRFI_bb_vs_co__raise, vsRFI_bb_vs_co__call, vsRFI_bn_vs_co__raise, vsRFI_sb_vs_co__raise, vsRFI_bn_vs_co__call, vsRFI_sb_vs_co__call, vsRFI_bb_vs_bn__raise, vsRFI_bb_vs_bn__call, vsRFI_sb_vs_bn__raise, vsRFI_bb_vs_sb__raise, vsRFI_bb_vs_sb__call]
opponents_ranges = [vsRFI_bb_vs_lj__raise, vsRFI_bb_vs_lj__call, vsRFI_hj_vs_lj__raise, vsRFI_co_vs_lj__raise, vsRFI_bn_vs_lj__raise, vsRFI_sb_vs_lj__raise, vsRFI_hj_vs_lj__call, vsRFI_co_vs_lj__call, vsRFI_bn_vs_lj__call, vsRFI_sb_vs_lj__call, vsRFI_bb_vs_hj__raise, vsRFI_bb_vs_hj__call, vsRFI_co_vs_hj__raise, vsRFI_bn_vs_hj__raise, vsRFI_sb_vs_hj__raise, vsRFI_co_vs_hj__call, vsRFI_bn_vs_hj__call, vsRFI_sb_vs_hj__call, vsRFI_bb_vs_co__raise, vsRFI_bb_vs_co__call, vsRFI_bn_vs_co__raise, vsRFI_sb_vs_co__raise, vsRFI_bn_vs_co__call, vsRFI_sb_vs_co__call, vsRFI_bb_vs_bn__raise, vsRFI_bb_vs_bn__call, vsRFI_sb_vs_bn__raise, vsRFI_bb_vs_sb__raise, vsRFI_bb_vs_sb__call, RFIvs3B_lj_vs_blinds_call, lj_rfi, RFIvs3B_lj_vs_hjco_call, RFIvs3B_lj_vs_hjco_call, RFIvs3B_lj_vs_bn_call, RFIvs3B_lj_vs_blinds_call, lj_rfi, lj_rfi, lj_rfi, lj_rfi, RFIvs3B_hj_vs_ahead_call, hj_rfi, RFIvs3B_hj_vs_ahead_call, RFIvs3B_hj_vs_ahead_call, RFIvs3B_hj_vs_ahead_call, hj_rfi, hj_rfi, hj_rfi, RFIvs3B_co_vs_blinds_call, co_rfi, RFIvs3B_co_vs_bn_call, RFIvs3B_co_vs_blinds_call, co_rfi, co_rfi, RFIvs3B_bnsb_vs_ahead_call, bn_rfi, RFIvs3B_bnsb_vs_ahead_call, RFIvs3B_bnsb_vs_ahead_call, sb_rfi]
my_position_ips = [True, True, False, False, False, True, False, False, False, True, True, True, False, False, True, False, False, True, True, True, False, True, False, True, True, True, True, False, False, False, False, True, True, True, False, True, True, True, False, False, False, True, True, False, True, True, False, False, False, True, False, True, False, False, False, False, True, True]
opponent_pfrs = [True, False, True, True, True, True, False, False, False, False, True, False, True, True, True, False, False, False, True, False, True, True, False, False, True, False, True, True, False, False, True, False, False, False, False, True, True, True, True, False, True, False, False, False, True, True, True, False, True, False, False, True, True, False, True, False, False, True]
pot_sizes = [18.5, 5.5, 19.5, 19.5, 19.5, 19, 6.5, 6.5, 6.5, 6, 18.5, 5.5, 19.5, 19.5, 19, 6.5, 6.5, 6, 18.5, 5.5, 19.5, 19, 6.5, 6, 18.5, 5.5, 19, 18, 6, 18.5, 5.5, 19.5, 19.5, 19.5, 19, 6.5, 6.5, 6.5, 6, 18.5, 5.5, 19.5, 19.5, 19, 6.5, 6.5, 6, 18.5, 5.5, 19.5, 19, 6.5, 6, 18.5, 5.5, 19, 18, 6]
my_investments = [9, 2.5, 9, 9, 9, 9, 2.5, 2.5, 2.5, 2.5, 9, 2.5, 9, 9, 9, 2.5, 2.5, 2.5, 9, 2.5, 9, 9, 2.5, 2.5, 9, 2.5, 9, 8.5, 2.5, 8, 1.5, 9, 9, 9, 8.5, 2.5, 2.5, 2.5, 2, 8, 1.5, 9, 9, 8.5, 2.5, 2.5, 2, 8, 1.5, 9, 8.5, 2.5, 2, 8, 1.5, 8.5, 8, 2]
print(len(range_names), len(my_ranges), len(opponents_ranges), len(my_position_ips), len(opponent_pfrs), len(pot_sizes), len(my_investments))




input = sys.argv[1]
if input == "1":
    board_type = "two-tone"
    start_index = 7
    range_names = range_names[start_index:]
    my_ranges = my_ranges[start_index:]
    opponents_ranges = opponents_ranges[start_index:]
    my_position_ips = my_position_ips[start_index:]
    opponent_pfrs = opponent_pfrs[start_index:]
    pot_sizes = pot_sizes[start_index:]
    my_investments = my_investments[start_index:]
elif input == "2":
    board_type = "rainbow"
    start_index = 2
    range_names = range_names[start_index:]
    my_ranges = my_ranges[start_index:]
    opponents_ranges = opponents_ranges[start_index:]
    my_position_ips = my_position_ips[start_index:]
    opponent_pfrs = opponent_pfrs[start_index:]
    pot_sizes = pot_sizes[start_index:]
    my_investments = my_investments[start_index:]
elif input == "3":
    board_type = "monotone"
    start_index = 11
    range_names = range_names[start_index:]
    my_ranges = my_ranges[start_index:]
    opponents_ranges = opponents_ranges[start_index:]
    my_position_ips = my_position_ips[start_index:]
    opponent_pfrs = opponent_pfrs[start_index:]
    pot_sizes = pot_sizes[start_index:]
    my_investments = my_investments[start_index:]
print(input, board_type)


# ***
for range_name, my_range, opponents_range, my_position_ip, opponent_pfr, pot_size, my_investment in zip(range_names, my_ranges, opponents_ranges, my_position_ips, opponent_pfrs, pot_sizes, my_investments):
    # Focus on one range for now, later iterate through all of them

    my_hands = range_to_hands(my_range)
    opponents_hands = range_to_hands(opponents_range)

    i = 0
    try:
        os.mkdir("../../reports/{}".format(range_name))
    except:
        pass
    try:
        os.mkdir("../../reports/{}/{}".format(range_name, board_type))
    except:
        pass



    for flop in flops:
        max_profit = -100
        last_profit = -100
        profits = {'flop': [], 'board_type': [], '0b': [], '1b': [], '2b': [], 'profit': []}


        # Focus on one flop for now, later iterate through all of them
        # flop = [11,6,3]

        # Interim variables
        is_paired = 1 if flop[0] == flop[1] or flop[1] == flop[2] else 0
        paired_value = 0 if not is_paired else flop[0] if flop[0] == flop[1] else flop[1]

        # Check for invalid flop + board type
        if board_type == "monotone" and is_paired:
            continue
        if board_type == "two-tone" and flop[0] == flop[2]:
            continue

        opponent_unraised_strategy = {
            'cat1': {1: 9, 2: 9, 3: 9, 4: 9, 5: 9, 6: 9, 7: 9},
            'cat2': {1: 2, 2: 5, 3: 6, 4: 6, 5: 7, 6: 7, 7: 7},
            'cat3': {1: 16 if opponent_pfr else 3,
                     2: 16 if opponent_pfr else 3,
                     3: 16 if opponent_pfr else 3,
                     4: 16 if opponent_pfr else 3,
                     5: 16 if opponent_pfr else 3,
                     6: 16 if opponent_pfr else 3,  # Opponent is way more bluff-heavy when the pfr; semi-good for them.
                     7: 16 if opponent_pfr else 3}, # skipping 4-10 for opponent's strategy (default in their strat)
        }
        opponent_strategy = {
            "oop_open": opponent_unraised_strategy,
            "oop_vs_cb": opponent_raised_strategy,
            "oop_vs_br": opponent_reraised_strategy,
            "ip_vs_c": opponent_unraised_strategy,
            "ip_vs_b": opponent_raised_strategy,
            "ip_vs_cbr": opponent_reraised_strategy,
        }





        # Important note: lower ranked rules may include higher ranked hands
            # Also tp_j_kicker includes trips because it's okay that it does because of the theory:
            # actions taken by one hand are taken by all better hands within the cat
            # ! Just be careful that you remove cat1 hands from final cat2, same with cat3 with both cat1 & 2
            # Might want to QA with a hand matrix coloring compare with the existing matrix based on default rules

        # Cat1
        #### Assuming no flushes (monotone boards) for simplicity
        my_hands_s_straight = [] if is_paired else [(x, 4) for x in my_hands[1] if max(x + flop) - min(x + flop) == 4 or max([1 if y == 14 else y for y in (x + flop)]) - min([1 if y == 14 else y for y in (x + flop)]) == 4]
        my_hands_o_straight = [] if is_paired else [(x, 12) for x in my_hands[2] if max(x + flop) - min(x + flop) == 4 or max([1 if y == 14 else y for y in (x + flop)]) - min([1 if y == 14 else y for y in (x + flop)]) == 4]
        my_hands_pp_sets = [(x, 3) for x in my_hands[0] if x[0] in flop]
        my_hands_s_trips = [] if not is_paired else [(x, 2) for x in my_hands[1] if x[0] == paired_value or x[1] == paired_value]
        my_hands_o_trips = [] if not is_paired else [(x, 6) for x in my_hands[2] if x[0] == paired_value or x[1] == paired_value]
        # 2 combos most times, not 3; 7 more often than 6
        my_hands_s_two_pair = [] if is_paired else [(x, 2) for x in my_hands[1] if x[0] in flop and x[1] in flop]
        my_hands_o_two_pair = [] if is_paired else [(x, 7) for x in my_hands[2] if x[0] in flop and x[1] in flop]
        my_hands_pp_overpair_9plus = [(x, 6) for x in my_hands[0] if x[0] > flop[0] and x[0] >= 9]
        my_hands_pp_any_overpair = [(x, 6) for x in my_hands[0] if x[0] > flop[0]]
        my_hands_s_tp_k_kicker = [(x, 3) for x in my_hands[1] if (x[0] == flop[0] and x[1] >= 13) or (x[1] == flop[0] and x[0] >= 13)]
        my_hands_o_tp_k_kicker = [(x, 9) for x in my_hands[2] if (x[0] == flop[0] and x[1] >= 13) or (x[1] == flop[0] and x[0] >= 13)]
        my_hands_s_tp_j_kicker = [(x, 3) for x in my_hands[1] if (x[0] == flop[0] and x[1] >= 11 and x[1] <= 12 and x[1] not in flop) or (x[1] == flop[0] and x[0] >= 11 and x[0] <= 12 and x[0] not in flop)]
        my_hands_o_tp_j_kicker = [(x, 9) for x in my_hands[2] if (x[0] == flop[0] and x[1] >= 11 and x[1] <= 12 and x[1] not in flop) or (x[1] == flop[0] and x[0] >= 11 and x[0] <= 12 and x[0] not in flop)]
        my_hands_s_tp_any_kicker = [(x, 3) for x in my_hands[1] if (x[0] == flop[0] and x[1] <= 10 and x[1] not in flop) or (x[1] == flop[0] and x[0] <= 10 and x[0] not in flop)]
        my_hands_o_tp_any_kicker = [(x, 9) for x in my_hands[2] if (x[0] == flop[0] and x[1] <= 10 and x[1] not in flop) or (x[1] == flop[0] and x[0] <= 10 and x[0] not in flop)]

        # Cat2 (flushdraws with high card hand might actually be part of cat3, but saying the combos are part of cat2)
        my_hands_s_tp_bad_kicker = [(x, 3) for x in my_hands[1] if (x[0] == flop[0] and x[1] <= 10 and x[1] not in flop) or (x[1] == flop[0] and x[0] <= 10 and x[0] not in flop)]
        my_hands_o_tp_bad_kicker = [(x, 9) for x in my_hands[2] if (x[0] == flop[0] and x[1] <= 10 and x[1] not in flop) or (x[1] == flop[0] and x[0] <= 10 and x[0] not in flop)]
        my_hands_s_middle_pair = [(x, 3) for x in my_hands[1] if (x[0] == flop[1] and x[1] not in flop) or (x[1] == flop[1] and x[0] not in flop)]
        my_hands_o_middle_pair = [(x, 9) for x in my_hands[2] if (x[0] == flop[1] and x[1] not in flop) or (x[1] == flop[1] and x[0] not in flop)]
        my_hands_s_bottom_pair = [(x, 3) for x in my_hands[1] if (x[0] == flop[2] and x[1] not in flop) or (x[1] == flop[2] and x[0] not in flop)]
        my_hands_o_bottom_pair = [(x, 9) for x in my_hands[2] if (x[0] == flop[2] and x[1] not in flop) or (x[1] == flop[2] and x[0] not in flop)]
        my_hands_pp_below_top_pair = [(x, 6) for x in my_hands[0] if x[0] < flop[0] and x[0] > flop[1]]
        my_hands_s_aj_high = [(x, 4) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and (x[0] == 14) and (x[1] > 10)]
        my_hands_o_aj_high = [(x, 12) for x in my_hands[2] if (x[0] not in flop and x[1] not in flop) and (x[0] == 14) and (x[1] > 10)]
        my_hands_pp_below_middle_pair = [(x, 6) for x in my_hands[0] if x[0] < flop[1] and x[0] > flop[2]]
        my_hands_s_kq_high = [(x, 4) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and ((x[0] == 13 and x[1] > 11) or (x[0] == 14))]
        my_hands_o_kq_high = [(x, 12) for x in my_hands[2] if (x[0] not in flop and x[1] not in flop) and ((x[0] == 13 and x[1] > 11) or (x[0] == 14))]
        my_hands_pp_below_bottom_pair = [(x, 6) for x in my_hands[0] if x[0] < flop[2]]
        my_hands_s_kj_high = [(x, 4) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and (x[0] == 13) and (x[1] == 11)]
        my_hands_o_kj_high = [(x, 12) for x in my_hands[2] if (x[0] not in flop and x[1] not in flop) and (x[0] == 13) and (x[1] == 11)]
        my_hands_s_k8_high = [(x, 4) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and (x[0] == 13) and (x[1] < 11 and x[1] >= 8)]
        my_hands_o_k8_high = [(x, 12) for x in my_hands[2] if (x[0] not in flop and x[1] not in flop) and (x[0] == 13) and (x[1] < 11 and x[1] >= 8)]

        # Cat3 paired
        #### bdfd/fd/no_fd combos should not be counted twice; combine them smartly and carefully.
        #### Assuming no suited flushdraw possible with pair on two-tone for simplicity
        #### Might say a straight is a gutshot but that is fine because future logic
        ## Include the combos for that hand
        if board_type == "two-tone":
            my_hands_pp_fd = []
            my_hands_s_fd = [(x, 1) for x in my_hands[1] if x[0] not in flop and x[1] not in flop]
            my_hands_o_fd = []
        elif board_type == "rainbow":
            my_hands_pp_fd = []
            my_hands_s_fd = []
            my_hands_o_fd = []
        else:
            my_hands_pp_fd = [(x, 3) for x in my_hands[0] if x[0] not in flop]
            my_hands_s_fd = []
            # If paired then ignore the flushdraw (anyway, just monotone; just makes things simpler)
            my_hands_o_fd = [(x, 6) for x in my_hands[2] if x[0] not in flop and x[1] not in flop]
        #### Also added double gutshots.  Doing a bit of overcounting for oesd+pair, which is fine for estimation (should be 9).
        my_hands_pp_oesd = [(x, 6) for x in my_hands[0] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 3) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 3) or (sorted(x+flop) == [3,4,5,7,14]) or (max(x+flop) - min(x+flop) == 6 and min(x+flop)+2 == sorted(x+flop)[1] and max(x+flop)-2 == sorted(x+flop)[-2])]
        my_hands_s_oesd = [(x, 4) for x in my_hands[1] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 3) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 3) or (sorted(x+flop) == [3,4,5,7,14]) or (max(x+flop) - min(x+flop) == 6 and min(x+flop)+2 == sorted(x+flop)[1] and max(x+flop)-2 == sorted(x+flop)[-2])]
        my_hands_o_oesd = [(x, 12) for x in my_hands[2] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 3) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 3) or (sorted(x+flop) == [3,4,5,7,14]) or (max(x+flop) - min(x+flop) == 6 and min(x+flop)+2 == sorted(x+flop)[1] and max(x+flop)-2 == sorted(x+flop)[-2])]
        my_hands_pp_gutshot = [(x, 6) for x in my_hands[0] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 4) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 4) or (sorted(set([1 if y == 14 else y for y in (x + flop)] + [20,21,22]))[3] <= 5)]
        my_hands_s_gutshot = [(x, 4) for x in my_hands[1] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 4) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 4) or (sorted(set([1 if y == 14 else y for y in (x + flop)] + [20,21,22]))[3] <= 5)]
        my_hands_o_gutshot = [(x, 12) for x in my_hands[2] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 4) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 4) or (sorted(set([1 if y == 14 else y for y in (x + flop)] + [20,21,22]))[3] <= 5)]
        #### Additional rule: 3 to a straight requires two cards from your hand not just one (that's how I want it to be)
        my_hands_s_3_to_straight_not_all_from_low_end = [(x, 4) for x in my_hands[1] if (x[0] != 14) and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]+1 and x[1] == flop[0]-1) or (x[0] == flop[0]+2 and x[1] == flop[0]+1) or (x[0] == flop[1]+1 and x[1] == flop[1]-1) or (x[0] == flop[1]+2 and x[1] == flop[1]+1) or (x[0] == flop[2]+1 and x[1] == flop[2]-1) or (x[0] == flop[2]+2 and x[1] == flop[2]+1))]
        my_hands_o_3_to_straight_not_all_from_low_end = [(x, 12) for x in my_hands[2] if (x[0] != 14) and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]+1 and x[1] == flop[0]-1) or (x[0] == flop[0]+2 and x[1] == flop[0]+1) or (x[0] == flop[1]+1 and x[1] == flop[1]-1) or (x[0] == flop[1]+2 and x[1] == flop[1]+1) or (x[0] == flop[2]+1 and x[1] == flop[2]-1) or (x[0] == flop[2]+2 and x[1] == flop[2]+1))]
        if board_type == "two-tone":
            my_hands_s_3_to_straight_low_end_bdfd = [(x, 1) for x in my_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            my_hands_o_3_to_straight_low_end_bdfd = [(x, 6) for x in my_hands[2] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
        elif board_type == "rainbow":
            my_hands_s_3_to_straight_low_end_bdfd = [(x, 3) for x in my_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            my_hands_o_3_to_straight_low_end_bdfd = []
        else:
            my_hands_s_3_to_straight_low_end_bdfd = []
            my_hands_o_3_to_straight_low_end_bdfd = []
        if board_type == "two-tone":
            my_hands_s_3_to_straight_low_end = [(x, 4) for x in my_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            my_hands_o_3_to_straight_low_end = [(x, 12) for x in my_hands[2] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
        elif board_type == "rainbow":
            my_hands_s_3_to_straight_low_end = [(x, 4) for x in my_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            my_hands_o_3_to_straight_low_end = [(x, 12) for x in my_hands[2] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
        else:
            my_hands_s_3_to_straight_low_end = [(x, 4) for x in my_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            my_hands_o_3_to_straight_low_end = [(x, 12) for x in my_hands[2] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
        if board_type == "two-tone":
            my_hands_s_5_unique_cards_within_7_values_bdfd = [(x, 1) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and max(flop+x) - min(flop+x) <= 7]
            my_hands_o_5_unique_cards_within_7_values_bdfd = [(x, 6) for x in my_hands[2] if (x[0] not in flop and x[1] not in flop) and max(flop+x) - min(flop+x) <= 7]
        elif board_type == "rainbow":
            my_hands_s_5_unique_cards_within_7_values_bdfd = [(x, 3) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and max(flop+x) - min(flop+x) <= 7]
            my_hands_o_5_unique_cards_within_7_values_bdfd = []
        else:
            my_hands_s_5_unique_cards_within_7_values_bdfd = []
            my_hands_o_5_unique_cards_within_7_values_bdfd = []
        if board_type == "two-tone":
            my_hands_pp_q_minus_bdfd = [(x, 3) for x in my_hands[0] if (x[0] not in flop) and x[0] <= 12]
            my_hands_s_q_minus_bdfd = [(x, 1) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and x[0] <= 12]
            my_hands_o_q_minus_bdfd = [(x, 6) for x in my_hands[2] if (x[0] not in flop and x[1] not in flop) and x[0] <= 12]
        elif board_type == "rainbow":
            my_hands_pp_q_minus_bdfd = []
            my_hands_s_q_minus_bdfd = [(x, 3) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and x[0] <= 12]
            my_hands_o_q_minus_bdfd = []
        else:
            my_hands_pp_q_minus_bdfd = []
            my_hands_s_q_minus_bdfd = []
            my_hands_o_q_minus_bdfd = []
        #### 3 cards within 4 values with two overcards
        my_hands_s_lowest_card_is_one_of_3_cards_within_4_values_and_two_overcards = [(x, 4) for x in my_hands[1] if x[1] > flop[0] and ((max(flop + [x[1]]) - sorted(set(flop + [x[1]] + [-20,-19,-18]))[-3] <= 3) or (max(flop + x) - sorted(set(flop + x + [-20,-19,-18]))[-3] <= 3))]
        my_hands_o_lowest_card_is_one_of_3_cards_within_4_values_and_two_overcards = [(x, 12) for x in my_hands[2] if x[1] > flop[0] and ((max(flop + [x[1]]) - sorted(set(flop + [x[1]] + [-20,-19,-18]))[-3] <= 3) or (max(flop + x) - sorted(set(flop + x + [-20,-19,-18]))[-3] <= 3))]
        if board_type == "two-tone":
            my_hands_pp_a_minus_bdfd = [(x, 3) for x in my_hands[0] if (x[0] not in flop) and x[0] > 12 and x[0] <= 14]
            my_hands_s_a_minus_bdfd = [(x, 1) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and x[0] > 12 and x[0] <= 14]
            my_hands_o_a_minus_bdfd = [(x, 6) for x in my_hands[2] if (x[0] not in flop and x[1] not in flop) and x[0] > 12 and x[0] <= 14]
        elif board_type == "rainbow":
            my_hands_pp_a_minus_bdfd = []
            my_hands_s_a_minus_bdfd = [(x, 3) for x in my_hands[1] if (x[0] not in flop and x[1] not in flop) and x[0] > 12 and x[0] <= 14]
            my_hands_o_a_minus_bdfd = []
        else:
            my_hands_pp_a_minus_bdfd = []
            my_hands_s_a_minus_bdfd = []
            my_hands_o_a_minus_bdfd = []




        # Important note: lower ranked rules may include higher ranked hands
            # Also tp_j_kicker includes trips because it's okay that it does because of the theory:
            # actions taken by one hand are taken by all better hands within the cat
            # ! Just be careful that you remove cat1 hands from final cat2, same with cat3 with both cat1 & 2
            # Might want to QA with a hand matrix coloring compare with the existing matrix based on default rules

        # Cat1
        #### Assuming no flushes (monotone boards) for simplicity
        opponents_hands_s_straight = [] if is_paired else [(x, 4) for x in opponents_hands[1] if max(x + flop) - min(x + flop) == 4 or max([1 if y == 14 else y for y in (x + flop)]) - min([1 if y == 14 else y for y in (x + flop)]) == 4]
        opponents_hands_o_straight = [] if is_paired else [(x, 12) for x in opponents_hands[2] if max(x + flop) - min(x + flop) == 4 or max([1 if y == 14 else y for y in (x + flop)]) - min([1 if y == 14 else y for y in (x + flop)]) == 4]
        opponents_hands_pp_sets = [(x, 3) for x in opponents_hands[0] if x[0] in flop]
        opponents_hands_s_trips = [] if not is_paired else [(x, 2) for x in opponents_hands[1] if x[0] == paired_value or x[1] == paired_value]
        opponents_hands_o_trips = [] if not is_paired else [(x, 6) for x in opponents_hands[2] if x[0] == paired_value or x[1] == paired_value]
        # 2 combos most times, not 3; 7 more often than 6
        opponents_hands_s_two_pair = [] if is_paired else [(x, 2) for x in opponents_hands[1] if x[0] in flop and x[1] in flop]
        opponents_hands_o_two_pair = [] if is_paired else [(x, 7) for x in opponents_hands[2] if x[0] in flop and x[1] in flop]
        opponents_hands_pp_overpair_9plus = [(x, 6) for x in opponents_hands[0] if x[0] > flop[0] and x[0] >= 9]
        opponents_hands_pp_any_overpair = [(x, 6) for x in opponents_hands[0] if x[0] > flop[0]]
        opponents_hands_s_tp_k_kicker = [(x, 3) for x in opponents_hands[1] if (x[0] == flop[0] and x[1] >= 13) or (x[1] == flop[0] and x[0] >= 13)]
        opponents_hands_o_tp_k_kicker = [(x, 9) for x in opponents_hands[2] if (x[0] == flop[0] and x[1] >= 13) or (x[1] == flop[0] and x[0] >= 13)]
        opponents_hands_s_tp_j_kicker = [(x, 3) for x in opponents_hands[1] if (x[0] == flop[0] and x[1] >= 11 and x[1] <= 12 and x[1] not in flop) or (x[1] == flop[0] and x[0] >= 11 and x[0] <= 12 and x[0] not in flop)]
        opponents_hands_o_tp_j_kicker = [(x, 9) for x in opponents_hands[2] if (x[0] == flop[0] and x[1] >= 11 and x[1] <= 12 and x[1] not in flop) or (x[1] == flop[0] and x[0] >= 11 and x[0] <= 12 and x[0] not in flop)]
        opponents_hands_s_tp_any_kicker = [(x, 3) for x in opponents_hands[1] if (x[0] == flop[0] and x[1] <= 10 and x[1] not in flop) or (x[1] == flop[0] and x[0] <= 10 and x[0] not in flop)]
        opponents_hands_o_tp_any_kicker = [(x, 9) for x in opponents_hands[2] if (x[0] == flop[0] and x[1] <= 10 and x[1] not in flop) or (x[1] == flop[0] and x[0] <= 10 and x[0] not in flop)]

        # Cat2 (flushdraws with high card hand might actually be part of cat3, but saying the combos are part of cat2)
        opponents_hands_s_tp_bad_kicker = [(x, 3) for x in opponents_hands[1] if (x[0] == flop[0] and x[1] <= 10 and x[1] not in flop) or (x[1] == flop[0] and x[0] <= 10 and x[0] not in flop)]
        opponents_hands_o_tp_bad_kicker = [(x, 9) for x in opponents_hands[2] if (x[0] == flop[0] and x[1] <= 10 and x[1] not in flop) or (x[1] == flop[0] and x[0] <= 10 and x[0] not in flop)]
        opponents_hands_s_middle_pair = [(x, 3) for x in opponents_hands[1] if (x[0] == flop[1] and x[1] not in flop) or (x[1] == flop[1] and x[0] not in flop)]
        opponents_hands_o_middle_pair = [(x, 9) for x in opponents_hands[2] if (x[0] == flop[1] and x[1] not in flop) or (x[1] == flop[1] and x[0] not in flop)]
        opponents_hands_s_bottom_pair = [(x, 3) for x in opponents_hands[1] if (x[0] == flop[2] and x[1] not in flop) or (x[1] == flop[2] and x[0] not in flop)]
        opponents_hands_o_bottom_pair = [(x, 9) for x in opponents_hands[2] if (x[0] == flop[2] and x[1] not in flop) or (x[1] == flop[2] and x[0] not in flop)]
        opponents_hands_pp_below_top_pair = [(x, 6) for x in opponents_hands[0] if x[0] < flop[0] and x[0] > flop[1]]
        opponents_hands_s_aj_high = [(x, 4) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and (x[0] == 14) and (x[1] > 10)]
        opponents_hands_o_aj_high = [(x, 12) for x in opponents_hands[2] if (x[0] not in flop and x[1] not in flop) and (x[0] == 14) and (x[1] > 10)]
        opponents_hands_pp_below_middle_pair = [(x, 6) for x in opponents_hands[0] if x[0] < flop[1] and x[0] > flop[2]]
        opponents_hands_s_kq_high = [(x, 4) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and ((x[0] == 13 and x[1] > 11) or (x[0] == 14))]
        opponents_hands_o_kq_high = [(x, 12) for x in opponents_hands[2] if (x[0] not in flop and x[1] not in flop) and ((x[0] == 13 and x[1] > 11) or (x[0] == 14))]
        opponents_hands_pp_below_bottom_pair = [(x, 6) for x in opponents_hands[0] if x[0] < flop[2]]
        opponents_hands_s_kj_high = [(x, 4) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and (x[0] == 13) and (x[1] == 11)]
        opponents_hands_o_kj_high = [(x, 12) for x in opponents_hands[2] if (x[0] not in flop and x[1] not in flop) and (x[0] == 13) and (x[1] == 11)]
        opponents_hands_s_k8_high = [(x, 4) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and (x[0] == 13) and (x[1] < 11 and x[1] >= 8)]
        opponents_hands_o_k8_high = [(x, 12) for x in opponents_hands[2] if (x[0] not in flop and x[1] not in flop) and (x[0] == 13) and (x[1] < 11 and x[1] >= 8)]

        # Cat3 paired
        #### bdfd/fd/no_fd combos should not be counted twice; combine them smartly and carefully.
        #### Assuming no suited flushdraw possible with pair on two-tone for simplicity
        #### Might say a straight is a gutshot but that is fine because future logic
        ## Include the combos for that hand
        if board_type == "two-tone":
            opponents_hands_pp_fd = []
            opponents_hands_s_fd = [(x, 1) for x in opponents_hands[1] if x[0] not in flop and x[1] not in flop]
            opponents_hands_o_fd = []
        elif board_type == "rainbow":
            opponents_hands_pp_fd = []
            opponents_hands_s_fd = []
            opponents_hands_o_fd = []
        else:
            opponents_hands_pp_fd = [(x, 3) for x in opponents_hands[0] if x[0] not in flop]
            opponents_hands_s_fd = []
            # If paired then ignore the flushdraw (anyway, just monotone; just makes things simpler)
            opponents_hands_o_fd = [(x, 6) for x in opponents_hands[2] if x[0] not in flop and x[1] not in flop]
        #### Also added double gutshots.  Doing a bit of overcounting for oesd+pair, which is fine for estimation (should be 9).
        opponents_hands_pp_oesd = [(x, 6) for x in opponents_hands[0] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 3) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 3) or (sorted(x+flop) == [3,4,5,7,14]) or (max(x+flop) - min(x+flop) == 6 and min(x+flop)+2 == sorted(x+flop)[1] and max(x+flop)-2 == sorted(x+flop)[-2])]
        opponents_hands_s_oesd = [(x, 4) for x in opponents_hands[1] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 3) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 3) or (sorted(x+flop) == [3,4,5,7,14]) or (max(x+flop) - min(x+flop) == 6 and min(x+flop)+2 == sorted(x+flop)[1] and max(x+flop)-2 == sorted(x+flop)[-2])]
        opponents_hands_o_oesd = [(x, 12) for x in opponents_hands[2] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 3) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 3) or (sorted(x+flop) == [3,4,5,7,14]) or (max(x+flop) - min(x+flop) == 6 and min(x+flop)+2 == sorted(x+flop)[1] and max(x+flop)-2 == sorted(x+flop)[-2])]
        opponents_hands_pp_gutshot = [(x, 6) for x in opponents_hands[0] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 4) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 4) or (sorted(set([1 if y == 14 else y for y in (x + flop)] + [20,21,22]))[3] <= 5)]
        opponents_hands_s_gutshot = [(x, 4) for x in opponents_hands[1] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 4) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 4) or (sorted(set([1 if y == 14 else y for y in (x + flop)] + [20,21,22]))[3] <= 5)]
        opponents_hands_o_gutshot = [(x, 12) for x in opponents_hands[2] if (sorted(set(x + flop + [20,21,22]))[3] - min(x + flop) == 4) or (max(x + flop) - sorted(set(x + flop + [-20,-19,-18]))[-4] == 4) or (sorted(set([1 if y == 14 else y for y in (x + flop)] + [20,21,22]))[3] <= 5)]
        #### Additional rule: 3 to a straight requires two cards from your hand not just one (that's how I want it to be)
        opponents_hands_s_3_to_straight_not_all_from_low_end = [(x, 4) for x in opponents_hands[1] if (x[0] != 14) and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]+1 and x[1] == flop[0]-1) or (x[0] == flop[0]+2 and x[1] == flop[0]+1) or (x[0] == flop[1]+1 and x[1] == flop[1]-1) or (x[0] == flop[1]+2 and x[1] == flop[1]+1) or (x[0] == flop[2]+1 and x[1] == flop[2]-1) or (x[0] == flop[2]+2 and x[1] == flop[2]+1))]
        opponents_hands_o_3_to_straight_not_all_from_low_end = [(x, 12) for x in opponents_hands[2] if (x[0] != 14) and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]+1 and x[1] == flop[0]-1) or (x[0] == flop[0]+2 and x[1] == flop[0]+1) or (x[0] == flop[1]+1 and x[1] == flop[1]-1) or (x[0] == flop[1]+2 and x[1] == flop[1]+1) or (x[0] == flop[2]+1 and x[1] == flop[2]-1) or (x[0] == flop[2]+2 and x[1] == flop[2]+1))]
        if board_type == "two-tone":
            opponents_hands_s_3_to_straight_low_end_bdfd = [(x, 1) for x in opponents_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            opponents_hands_o_3_to_straight_low_end_bdfd = [(x, 6) for x in opponents_hands[2] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
        elif board_type == "rainbow":
            opponents_hands_s_3_to_straight_low_end_bdfd = [(x, 3) for x in opponents_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            opponents_hands_o_3_to_straight_low_end_bdfd = []
        else:
            opponents_hands_s_3_to_straight_low_end_bdfd = []
            opponents_hands_o_3_to_straight_low_end_bdfd = []
        if board_type == "two-tone":
            opponents_hands_s_3_to_straight_low_end = [(x, 4) for x in opponents_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            opponents_hands_o_3_to_straight_low_end = [(x, 12) for x in opponents_hands[2] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
        elif board_type == "rainbow":
            opponents_hands_s_3_to_straight_low_end = [(x, 4) for x in opponents_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            opponents_hands_o_3_to_straight_low_end = [(x, 12) for x in opponents_hands[2] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
        else:
            opponents_hands_s_3_to_straight_low_end = [(x, 4) for x in opponents_hands[1] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
            opponents_hands_o_3_to_straight_low_end = [(x, 12) for x in opponents_hands[2] if x[0] != 13 and (x[0] not in flop and x[1] not in flop) and ((x[0] == flop[0]-1 and x[1] == flop[0]-2) or (x[0] == flop[1]-1 and x[1] == flop[1]-2) or (x[0] == flop[2]-1 and x[1] == flop[2]-2))]
        if board_type == "two-tone":
            opponents_hands_s_5_unique_cards_within_7_values_bdfd = [(x, 1) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and max(flop+x) - min(flop+x) <= 7]
            opponents_hands_o_5_unique_cards_within_7_values_bdfd = [(x, 6) for x in opponents_hands[2] if (x[0] not in flop and x[1] not in flop) and max(flop+x) - min(flop+x) <= 7]
        elif board_type == "rainbow":
            opponents_hands_s_5_unique_cards_within_7_values_bdfd = [(x, 3) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and max(flop+x) - min(flop+x) <= 7]
            opponents_hands_o_5_unique_cards_within_7_values_bdfd = []
        else:
            opponents_hands_s_5_unique_cards_within_7_values_bdfd = []
            opponents_hands_o_5_unique_cards_within_7_values_bdfd = []
        if board_type == "two-tone":
            opponents_hands_pp_q_minus_bdfd = [(x, 3) for x in opponents_hands[0] if (x[0] not in flop) and x[0] <= 12]
            opponents_hands_s_q_minus_bdfd = [(x, 1) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and x[0] <= 12]
            opponents_hands_o_q_minus_bdfd = [(x, 6) for x in opponents_hands[2] if (x[0] not in flop and x[1] not in flop) and x[0] <= 12]
        elif board_type == "rainbow":
            opponents_hands_pp_q_minus_bdfd = []
            opponents_hands_s_q_minus_bdfd = [(x, 3) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and x[0] <= 12]
            opponents_hands_o_q_minus_bdfd = []
        else:
            opponents_hands_pp_q_minus_bdfd = []
            opponents_hands_s_q_minus_bdfd = []
            opponents_hands_o_q_minus_bdfd = []
        #### 3 cards within 4 values with two overcards
        opponents_hands_s_lowest_card_is_one_of_3_cards_within_4_values_and_two_overcards = [(x, 4) for x in opponents_hands[1] if x[1] > flop[0] and ((max(flop + [x[1]]) - sorted(set(flop + [x[1]] + [-20,-19,-18]))[-3] <= 3) or (max(flop + x) - sorted(set(flop + x + [-20,-19,-18]))[-3] <= 3))]
        opponents_hands_o_lowest_card_is_one_of_3_cards_within_4_values_and_two_overcards = [(x, 12) for x in opponents_hands[2] if x[1] > flop[0] and ((max(flop + [x[1]]) - sorted(set(flop + [x[1]] + [-20,-19,-18]))[-3] <= 3) or (max(flop + x) - sorted(set(flop + x + [-20,-19,-18]))[-3] <= 3))]
        if board_type == "two-tone":
            opponents_hands_pp_a_minus_bdfd = [(x, 3) for x in opponents_hands[0] if (x[0] not in flop) and x[0] > 12 and x[0] <= 14]
            opponents_hands_s_a_minus_bdfd = [(x, 1) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and x[0] > 12 and x[0] <= 14]
            opponents_hands_o_a_minus_bdfd = [(x, 6) for x in opponents_hands[2] if (x[0] not in flop and x[1] not in flop) and x[0] > 12 and x[0] <= 14]
        elif board_type == "rainbow":
            opponents_hands_pp_a_minus_bdfd = []
            opponents_hands_s_a_minus_bdfd = [(x, 3) for x in opponents_hands[1] if (x[0] not in flop and x[1] not in flop) and x[0] > 12 and x[0] <= 14]
            opponents_hands_o_a_minus_bdfd = []
        else:
            opponents_hands_pp_a_minus_bdfd = []
            opponents_hands_s_a_minus_bdfd = []
            opponents_hands_o_a_minus_bdfd = []



        def main(cat3_level_0b, cat3_level_1b, cat3_level_2b, max_profit, last_profit, profits, profit_became_worse, profit_became_worse_twice):

            # Determine my strategy this hand
            my_hands_cat1_0b = my_hands_cat1_level_x_and_above(8)  # Always 8

            # If IP + (flop type 6 or 7 or PFR)
            flop_type_number = get_flop_type_number()
            if my_position_ip and (flop_type_number >= 6 or not opponent_pfr):
                my_hands_cat2_0b = my_hands_cat2_level_x_and_above(7, my_hands_cat1_0b)
            else:
                my_hands_cat2_0b = my_hands_cat2_level_x_and_above(5, my_hands_cat1_0b)
            my_hands_cat3_0b = None


            # # Removed because these constraints are unprofitable as of now
            # # Interim step (50:50 to 30:70 ratio cat1 to cat3)
            # cat3_combos_0b_lb = sum([y for (x,y) in my_hands_cat1_0b[0]])+sum([y for (x,y) in my_hands_cat1_0b[1]])+sum([y for (x,y) in my_hands_cat1_0b[2]])
            # cat3_combos_0b_ub = cat3_combos_0b_lb * 2.3
            #
            # # Initialize
            # my_cat3_level_lb = 1
            # my_cat3_level_ub = 21
            #
            # # Find bounds
            # cat3_combos = 0
            # for i in range(20):
            #     if cat3_combos >= cat3_combos_0b_lb:
            #         break
            #     else:
            #         my_cat3_level_lb += 1
            #     my_hands_cat3_0b = my_hands_cat3_level_x_and_above(my_cat3_level_lb, my_hands_cat1_0b, my_hands_cat2_0b)
            #     cat3_combos = sum([y for (x,y) in my_hands_cat3_0b[0]])+sum([y for (x,y) in my_hands_cat3_0b[1]])+sum([y for (x,y) in my_hands_cat3_0b[2]])
            #
            # cat3_combos = 400
            # for i in range(20):
            #     if cat3_combos <= cat3_combos_0b_ub:
            #         break
            #     else:
            #         my_cat3_level_ub -= 1
            #     my_hands_cat3_0b = my_hands_cat3_level_x_and_above(my_cat3_level_ub, my_hands_cat1_0b, my_hands_cat2_0b)
            #     cat3_combos = sum([y for (x,y) in my_hands_cat3_0b[0]])+sum([y for (x,y) in my_hands_cat3_0b[1]])+sum([y for (x,y) in my_hands_cat3_0b[2]])

            # Print bounds
            # print("0b: my_cat3_level_lb", my_cat3_level_lb)
            # print("0b: my_cat3_level_ub", my_cat3_level_ub)

            my_hands_cat3_0b = my_hands_cat3_level_x_and_above(cat3_level_0b, my_hands_cat1_0b, my_hands_cat2_0b)





            # Determine my strategy this hand
            my_hands_cat1_1b = my_hands_cat1_level_x_and_above(6)
            if my_position_ip and (flop_type_number >= 6 or not opponent_pfr):
                my_hands_cat2_1b = my_hands_cat2_level_x_and_above(7, my_hands_cat1_1b)
            else:
                my_hands_cat2_1b = my_hands_cat2_level_x_and_above(5, my_hands_cat1_1b)
            my_hands_cat3_1b = None

            # # Interim step (50:50 to 30:70 ratio cat1 to cat3)
            # cat3_combos_1b_lb = sum([y for (x,y) in my_hands_cat1_1b[0]])+sum([y for (x,y) in my_hands_cat1_1b[1]])+sum([y for (x,y) in my_hands_cat1_1b[2]])
            # cat3_combos_1b_ub = cat3_combos_1b_lb * 2.3
            #
            # # Initialize
            # my_cat3_level_lb = 1
            # my_cat3_level_ub = 21
            #
            # # Find bounds
            # cat3_combos = 0
            # for i in range(20):
            #     if cat3_combos >= cat3_combos_1b_lb:
            #         break
            #     else:
            #         my_cat3_level_lb += 1
            #     my_hands_cat3_1b = my_hands_cat3_level_x_and_above(my_cat3_level_lb, my_hands_cat1_1b, my_hands_cat2_1b)
            #     cat3_combos = sum([y for (x,y) in my_hands_cat3_1b[0]])+sum([y for (x,y) in my_hands_cat3_1b[1]])+sum([y for (x,y) in my_hands_cat3_1b[2]])
            #
            # cat3_combos = 400
            # for i in range(20):
            #     if cat3_combos <= cat3_combos_1b_ub:
            #         break
            #     else:
            #         my_cat3_level_ub -= 1
            #     my_hands_cat3_1b = my_hands_cat3_level_x_and_above(my_cat3_level_ub, my_hands_cat1_1b, my_hands_cat2_1b)
            #     cat3_combos = sum([y for (x,y) in my_hands_cat3_1b[0]])+sum([y for (x,y) in my_hands_cat3_1b[1]])+sum([y for (x,y) in my_hands_cat3_1b[2]])

            # Print bounds
            # print("1b: my_cat3_level_lb", my_cat3_level_lb)
            # print("1b: my_cat3_level_ub", my_cat3_level_ub)

            my_hands_cat3_1b = my_hands_cat3_level_x_and_above(cat3_level_1b, my_hands_cat1_1b, my_hands_cat2_1b)






            # Determine my strategy this hand
            my_hands_cat1_2b = my_hands_cat1_level_x_and_above(0)
            my_hands_cat2_2b = my_hands_cat2_level_x_and_above(3, my_hands_cat1_2b)
            my_hands_cat3_2b = None

            # Interim step (50:50 to 30:70 ratio cat1 to cat3)
            # cat3_combos_2b_lb = sum([y for (x,y) in my_hands_cat1_2b[0]])+sum([y for (x,y) in my_hands_cat1_2b[1]])+sum([y for (x,y) in my_hands_cat1_2b[2]])
            # cat3_combos_2b_ub = cat3_combos_2b_lb * 2.3
            #
            # # Initialize
            # my_cat3_level_lb = 1
            # my_cat3_level_ub = 21  # Can lower this to something like 10 or lower once experimenting on a few hands
            # # We are just calling with our cat3 hands

            # Print bounds
            # print("2b: my_cat3_level_lb", my_cat3_level_lb)
            # print("2b: my_cat3_level_ub", my_cat3_level_ub)





            my_hands_cat3_2b = my_hands_cat3_level_x_and_above(cat3_level_2b, my_hands_cat1_2b, my_hands_cat2_2b)






            # Changing variable
            bets_so_far = 0 # might delete this variable

            # Determine my bet size
            my_bet_size = \
                0.4 if is_paired else \
                0.6 if board_type == "rainbow" and flop[0] - flop[1] >= 5 else \
                0.8 if board_type != "rainbow" and flop[0] - flop[1] <= 4 else \
                0.7

            # Determine my opponent's strategy this hand
            #### Identify which of the 8 flops this is to decide the opponent's strategy they will use
            opponents_hands_cat1_0b = opponents_hands_cat1_level_x_and_above(opponent_strategy[get_opponent_situation(0)]["cat1"][flop_type_number])
            opponents_hands_cat2_0b = opponents_hands_cat2_level_x_and_above(min(7, opponent_strategy[get_opponent_situation(0)]["cat2"][flop_type_number] + 1 if not my_position_ip else 0), opponents_hands_cat1_0b) # they are wider IP, but not wider than 7
            opponents_hands_cat3_0b = opponents_hands_cat3_level_x_and_above(opponent_strategy[get_opponent_situation(0)]["cat3"][flop_type_number] + 1 if not my_position_ip else 0, opponents_hands_cat1_0b, opponents_hands_cat2_0b) # they are wider IP
            opponents_hands_cat1_1b = opponents_hands_cat1_level_x_and_above(opponent_strategy[get_opponent_situation(1)]["cat1"][flop_type_number])
            opponents_hands_cat2_1b = opponents_hands_cat2_level_x_and_above(min(7, opponent_strategy[get_opponent_situation(1)]["cat2"][flop_type_number] + 1 if not my_position_ip else 0), opponents_hands_cat1_1b) # they are wider IP, but not wider than 7
            opponents_hands_cat3_1b = opponents_hands_cat3_level_x_and_above(opponent_strategy[get_opponent_situation(1)]["cat3"][flop_type_number] + 1 if not my_position_ip else 0, opponents_hands_cat1_1b, opponents_hands_cat2_1b) # they are wider IP
            opponents_hands_cat1_2b = opponents_hands_cat1_level_x_and_above(opponent_strategy[get_opponent_situation(2)]["cat1"][flop_type_number])
            opponents_hands_cat2_2b = opponents_hands_cat2_level_x_and_above(min(7, opponent_strategy[get_opponent_situation(2)]["cat2"][flop_type_number] + 1 if not my_position_ip else 0), opponents_hands_cat1_2b) # they are wider IP, but not wider than 7
            opponents_hands_cat3_2b = opponents_hands_cat3_level_x_and_above(opponent_strategy[get_opponent_situation(2)]["cat3"][flop_type_number] + 1 if not my_position_ip else 0, opponents_hands_cat1_2b, opponents_hands_cat2_2b) # they are wider IP

            # Determine opponent's bet size:
            opponents_bet_size = 0.60





            opponents_hands_with_combos = [[], [], []]
            opponents_hands_with_combos[0] = [(x, 6) if x[0] not in flop else (x, 3) for x in opponents_hands[0]]
            opponents_hands_with_combos[1] = [(x, 4) if x[0] not in flop and x[1] not in flop else (x, 2) if x[0] in flop and x[1] in flop else (x,3) for x in opponents_hands[1]]
            opponents_hands_with_combos[2] = [(x, 12) if x[0] not in flop and x[1] not in flop else (x, 7) if x[0] in flop and x[1] in flop else (x,9) for x in opponents_hands[2]]

            def count_hand_combos(hands):
                return sum([y for (x,y) in hands[0]])+sum([y for (x,y) in hands[1]])+sum([y for (x,y) in hands[2]])

            def get_check_hands(all_hands_before_action_w_combos, cat1_hands_for_action, cat3_hands_for_action):
                hands = [[], [], []]
                temp_cat3_hands = deepcopy(cat3_hands_for_action)

                # Flip sign (for subtraction)
                for i in range(3):
                    temp_cat3_hands[i] = [(x, -1*y) for (x, y) in temp_cat3_hands[i]]

                # Combine (for subtraction)
                result = combine_hands(all_hands_before_action_w_combos, temp_cat3_hands)

                # Subtraction
                for i in range(3):
                    groupby_dict = defaultdict(int)
                    for val in result[i]:
                        groupby_dict[tuple(val[0])] += val[1]
                    result[i] = [(sorted(list(x), reverse=True), max(0, min(y, 6 if i == 0 else 4 if i == 1 else 12))) for (x,y) in groupby_dict.items()]
                    result[i] = [(x,y) for (x,y) in result[i] if y != 0 and x not in [x for (x,y) in cat1_hands_for_action[i]]]
                return result

            def get_fold_hands(all_hands_before_action_w_combos, cat1_hands_for_action, cat2_hands_for_action, cat3_hands_for_action):
                hands = get_check_hands(all_hands_before_action_w_combos, cat1_hands_for_action, cat3_hands_for_action)
                for i in range(3):
                    hands[i] = [(x,y) for (x,y) in hands[i] if x not in [x for (x,y) in cat2_hands_for_action[i]]]
                return hands

            def get_call_hands(all_hands_before_action_w_combos, cat2_hands_for_action):
                hands = deepcopy(cat2_hands_for_action)
                for i in range(3):
                    hands[i] = [(x,y) for (x,y) in hands[i] if x in [x for (x,y) in all_hands_before_action_w_combos[i]]]
                return hands

            def get_raise_hands(all_hands_before_action_w_combos, cat1_hands_for_action, cat3_hands_for_action):
                hands = combine_hands(cat1_hands_for_action, cat3_hands_for_action)
                for i in range(3):
                    hands[i] = [(x,y) for (x,y) in hands[i] if x in [x for (x,y) in all_hands_before_action_w_combos[i]]]
                return hands

            def combine_hands(hands1, hands2):
                hands = [[], [], []]
                for i in range(3):
                    hands[i] = hands1[i] + hands2[i]
                return hands




            my_hands_with_combos = [[], [], []]
            my_hands_with_combos[0] = [(x, 6) if x[0] not in flop else (x, 3) for x in my_hands[0]]
            my_hands_with_combos[1] = [(x, 4) if x[0] not in flop and x[1] not in flop else (x, 2) if x[0] in flop and x[1] in flop else (x,3) for x in my_hands[1]]
            my_hands_with_combos[2] = [(x, 12) if x[0] not in flop and x[1] not in flop else (x, 7) if x[0] in flop and x[1] in flop else (x,9) for x in my_hands[2]]







            # Determine:
            #### 0) Hands in each situation
            if my_position_ip:
                # Hands
                opponents_hands_c = get_check_hands(opponents_hands_with_combos, opponents_hands_cat1_0b, opponents_hands_cat3_0b)
                opponents_hands_b = combine_hands(opponents_hands_cat1_0b, opponents_hands_cat3_0b)
                my_hands_cc = get_check_hands(my_hands_with_combos, my_hands_cat1_0b, my_hands_cat3_0b)
                my_hands_cb = combine_hands(my_hands_cat1_0b, my_hands_cat3_0b)
                my_hands_bf = get_fold_hands(my_hands_with_combos, my_hands_cat1_1b, my_hands_cat2_1b, my_hands_cat3_1b)
                my_hands_bc = my_hands_cat2_1b
                my_hands_bb = combine_hands(my_hands_cat1_1b, my_hands_cat3_1b)
                opponents_hands_cbf = get_fold_hands(opponents_hands_c, opponents_hands_cat1_1b, opponents_hands_cat2_1b, opponents_hands_cat3_1b)
                opponents_hands_cbc = get_call_hands(opponents_hands_c, opponents_hands_cat2_1b)
                opponents_hands_cbb = get_raise_hands(opponents_hands_c, opponents_hands_cat1_1b, opponents_hands_cat3_1b)
                opponents_hands_bbf = get_fold_hands(opponents_hands_b, opponents_hands_cat1_2b, opponents_hands_cat2_2b, opponents_hands_cat3_2b)
                opponents_hands_bbc = get_call_hands(opponents_hands_b, combine_hands(opponents_hands_cat1_2b, combine_hands(opponents_hands_cat2_2b, opponents_hands_cat3_2b))) # cat1/3 are calls
                my_hands_cbbf = get_fold_hands(my_hands_cb, my_hands_cat1_2b, my_hands_cat2_2b, my_hands_cat3_2b)
                my_hands_cbbc = get_call_hands(my_hands_cb, combine_hands(my_hands_cat1_2b, combine_hands(my_hands_cat2_2b, my_hands_cat3_2b))) # cat1/3 are calls

                # Combos
                combos_c = count_hand_combos(opponents_hands_c)
                combos_b = count_hand_combos(opponents_hands_b)
                combos_cc = count_hand_combos(my_hands_cc)
                combos_cb = count_hand_combos(my_hands_cb)
                combos_bf = count_hand_combos(my_hands_bf)
                combos_bc = count_hand_combos(my_hands_bc)
                combos_bb = count_hand_combos(my_hands_bb)
                combos_cbf = count_hand_combos(opponents_hands_cbf)
                combos_cbc = count_hand_combos(opponents_hands_cbc)
                combos_cbb = count_hand_combos(opponents_hands_cbb)
                combos_bbf = count_hand_combos(opponents_hands_bbf)
                combos_bbc = count_hand_combos(opponents_hands_bbc)
                combos_cbbf = count_hand_combos(my_hands_cbbf)
                combos_cbbc = count_hand_combos(my_hands_cbbc)

                # Cat3 pct_makeup
                opponents_cat3_pct_cc = 0
                opponents_cat3_pct_cbc = 0
                opponents_cat3_pct_cbbc = 0 if combos_cbb == 0 else count_hand_combos(get_raise_hands(opponents_hands_c, [[],[],[]], opponents_hands_cat3_1b)) / combos_cbb
                opponents_cat3_pct_bc = 0 if combos_b == 0 else count_hand_combos(opponents_hands_cat3_0b) / combos_b
                opponents_cat3_pct_bbc = 0 if combos_bbc == 0 else count_hand_combos(get_raise_hands(opponents_hands_b, [[],[],[]], opponents_hands_cat3_2b)) / combos_bbc

                my_cat3_pct_cc = 0
                my_cat3_pct_cbc = 0 if combos_cb == 0 else count_hand_combos(my_hands_cat3_0b) / combos_cb
                my_cat3_pct_cbbc = 0 if combos_cbbc == 0 else count_hand_combos(get_raise_hands(my_hands_cb, [[],[],[]], my_hands_cat3_2b)) / combos_cbbc
                my_cat3_pct_bc = 0
                my_cat3_pct_bbc = 0 if combos_bb == 0 else count_hand_combos(my_hands_cat3_1b) / combos_bb






            else:
                # Hands
                my_hands_c = get_check_hands(my_hands_with_combos, my_hands_cat1_0b, my_hands_cat3_0b)
                my_hands_b = combine_hands(my_hands_cat1_0b, my_hands_cat3_0b)
                opponents_hands_cc = get_check_hands(opponents_hands_with_combos, opponents_hands_cat1_0b, opponents_hands_cat3_0b)
                opponents_hands_cb = combine_hands(opponents_hands_cat1_0b, opponents_hands_cat3_0b)
                opponents_hands_bf = get_fold_hands(opponents_hands_with_combos, opponents_hands_cat1_1b, opponents_hands_cat2_1b, opponents_hands_cat3_1b)
                opponents_hands_bc = opponents_hands_cat2_1b
                opponents_hands_bb = combine_hands(opponents_hands_cat1_1b, opponents_hands_cat3_1b)
                my_hands_cbf = get_fold_hands(my_hands_c, my_hands_cat1_1b, my_hands_cat2_1b, my_hands_cat3_1b)
                my_hands_cbc = get_call_hands(my_hands_c, my_hands_cat2_1b)
                my_hands_cbb = get_raise_hands(my_hands_c, my_hands_cat1_1b, my_hands_cat3_1b)
                my_hands_bbf = get_fold_hands(my_hands_b, my_hands_cat1_2b, my_hands_cat2_2b, my_hands_cat3_2b)
                my_hands_bbc = get_call_hands(my_hands_b, combine_hands(my_hands_cat1_2b, combine_hands(my_hands_cat2_2b, my_hands_cat3_2b))) # cat1/3 are calls
                opponents_hands_cbbf = get_fold_hands(opponents_hands_cb, opponents_hands_cat1_2b, opponents_hands_cat2_2b, opponents_hands_cat3_2b)
                opponents_hands_cbbc = get_call_hands(opponents_hands_cb, combine_hands(opponents_hands_cat1_2b, combine_hands(opponents_hands_cat2_2b, opponents_hands_cat3_2b))) # cat1/3 are calls

                # Combos
                combos_c = count_hand_combos(my_hands_c)
                combos_b = count_hand_combos(my_hands_b)
                combos_cc = count_hand_combos(opponents_hands_cc)
                combos_cb = count_hand_combos(opponents_hands_cb)
                combos_bf = count_hand_combos(opponents_hands_bf)
                combos_bc = count_hand_combos(opponents_hands_bc)
                combos_bb = count_hand_combos(opponents_hands_bb)
                combos_cbf = count_hand_combos(my_hands_cbf)
                combos_cbc = count_hand_combos(my_hands_cbc)
                combos_cbb = count_hand_combos(my_hands_cbb)
                combos_bbf = count_hand_combos(my_hands_bbf)
                combos_bbc = count_hand_combos(my_hands_bbc)
                combos_cbbf = count_hand_combos(opponents_hands_cbbf)
                combos_cbbc = count_hand_combos(opponents_hands_cbbc)

                # Cat3 pct_makeup
                my_cat3_pct_cc = 0
                my_cat3_pct_cbc = 0
                my_cat3_pct_cbbc = 0 if combos_cbb == 0 else count_hand_combos(get_raise_hands(my_hands_c, [[],[],[]], my_hands_cat3_1b)) / combos_cbb
                my_cat3_pct_bc = 0 if combos_b == 0 else count_hand_combos(my_hands_cat3_0b) / combos_b
                my_cat3_pct_bbc = 0 if combos_bbc == 0 else count_hand_combos(get_raise_hands(my_hands_b, [[],[],[]], my_hands_cat3_2b)) / combos_bbc

                opponents_cat3_pct_cc = 0
                opponents_cat3_pct_cbc = 0 if combos_cb == 0 else count_hand_combos(opponents_hands_cat3_0b) / combos_cb
                opponents_cat3_pct_cbbc = 0 if combos_cbbc == 0 else count_hand_combos(get_raise_hands(opponents_hands_cb, [[],[],[]], opponents_hands_cat3_2b)) / combos_cbbc
                opponents_cat3_pct_bc = 0
                opponents_cat3_pct_bbc = 0 if combos_bb == 0 else count_hand_combos(opponents_hands_cat3_1b) / combos_bb


            #### 1) the % chance of each bet sequence

            chance_c = 0 if (combos_c + combos_b) == 0 else combos_c/(combos_c + combos_b)
            chance_b = 0 if (combos_c + combos_b) == 0 else combos_b/(combos_c + combos_b)
            chance_cc = 0 if (combos_cc + combos_cb) == 0 else chance_c*(combos_cc/(combos_cc + combos_cb))
            chance_cb = 0 if (combos_cc + combos_cb) == 0 else chance_c*(combos_cb/(combos_cc + combos_cb))
            chance_bf = 0 if (combos_bf + combos_bc + combos_bb) == 0 else chance_b*(combos_bf/(combos_bf + combos_bc + combos_bb))
            chance_bc = 0 if (combos_bf + combos_bc + combos_bb) == 0 else chance_b*(combos_bc/(combos_bf + combos_bc + combos_bb))
            chance_bb = 0 if (combos_bf + combos_bc + combos_bb) == 0 else chance_b*(combos_bb/(combos_bf + combos_bc + combos_bb))
            chance_cbf = 0 if (combos_cbf + combos_cbc + combos_cbb) == 0 else chance_cb*(combos_cbf/(combos_cbf + combos_cbc + combos_cbb))
            chance_cbc = 0 if (combos_cbf + combos_cbc + combos_cbb) == 0 else chance_cb*(combos_cbc/(combos_cbf + combos_cbc + combos_cbb))
            chance_cbb = 0 if (combos_cbf + combos_cbc + combos_cbb) == 0 else chance_cb*(combos_cbb/(combos_cbf + combos_cbc + combos_cbb))
            chance_bbf = 0 if (combos_bbf + combos_bbc) == 0 else chance_bb*(combos_bbf/(combos_bbf + combos_bbc))
            chance_bbc = 0 if (combos_bbf + combos_bbc) == 0 else chance_bb*(combos_bbc/(combos_bbf + combos_bbc))
            chance_cbbf = 0 if (combos_cbbf + combos_cbbc) == 0 else chance_cbb*(combos_cbbf/(combos_cbbf + combos_cbbc))
            chance_cbbc = 0 if (combos_cbbf + combos_cbbc) == 0 else chance_cbb*(combos_cbbc/(combos_cbbf + combos_cbbc))

            # print("Test that all add to 1.0")
            chance_c+chance_b, chance_cc+chance_cb+chance_bf+chance_bc+chance_bb, \
            chance_cc+chance_cbf+chance_cbc+chance_cbb+chance_bf+chance_bc+chance_bbf+chance_bbc, \
            chance_cc+chance_cbf+chance_cbc+chance_cbbf+chance_cbbc+chance_bf+chance_bc+chance_bbf+chance_bbc





            m2 = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
                  9: '9', 8: '8', 7: '7', 6: '6', 5: '5',
                  4: '4', 3: '3', 2: '2'}
            def convert_hands_to_range(hands, flop):
                result = []
                for hand in hands[0]:
                    rank = hand[0][0]
                    if hand[1] != 6 and rank not in flop:
                        result.append(m2[rank] + "s" + m2[rank] + "h")
                        result.append(m2[rank] + "s" + m2[rank] + "c")
                        result.append(m2[rank] + "s" + m2[rank] + "d")
                    else:
                        result.append(m2[rank]*2)
                for hand in hands[1]:
                    rank1 = hand[0][0]
                    rank2 = hand[0][1]
                    if hand[1] == 4:
                        result.append(m2[rank1] + m2[rank2])
                    elif hand[1] == 3 and (rank1 in flop or rank2 in flop):
                        result.append(m2[rank1] + m2[rank2])
                    elif hand[1] == 3:
                        result.append(m2[rank1] + "s" + m2[rank2] + "s")
                        result.append(m2[rank1] + "h" + m2[rank2] + "h")
                        result.append(m2[rank1] + "c" + m2[rank2] + "c")
                    elif hand[1] == 2 and ((rank1 in flop and rank2 in flop) or (rank1 == paired_value or rank2 == paired_value)):
                        result.append(m2[rank1] + m2[rank2])
                    elif hand[1] == 2:
                        result.append(m2[rank1] + "s" + m2[rank2] + "s")
                        result.append(m2[rank1] + "h" + m2[rank2] + "h")
                    elif hand[1] == 1:
                        result.append(m2[rank1] + "s" + m2[rank2] + "s")
                for hand in hands[2]:
                    rank1 = hand[0][0]
                    rank2 = hand[0][1]
                    if hand[1] == 12:
                        result.append(m2[rank1] + m2[rank2])
                    elif rank1 in flop or rank2 in flop:
                        result.append(m2[rank1] + m2[rank2])
                    elif hand[1] == 6:
                        # one spade
                        result.append(m2[rank1] + "s" + m2[rank2] + "h")
                        result.append(m2[rank1] + "s" + m2[rank2] + "c")
                        result.append(m2[rank1] + "s" + m2[rank2] + "d")
                        result.append(m2[rank1] + "h" + m2[rank2] + "s")
                        result.append(m2[rank1] + "c" + m2[rank2] + "s")
                        result.append(m2[rank1] + "d" + m2[rank2] + "s")
                    else:
                        raise Exception # Should never occur, investigate if this occurs
                return ",".join(result)



            #### 2) the ranges that go against each other (or who won pot)
            if my_position_ip:
                # Hands version
                final_opponents_hands_cc = opponents_hands_c
                final_my_hands_cc = my_hands_cc
                final_opponents_hands_cbc = opponents_hands_cbc
                final_my_hands_cbc = my_hands_cb
                final_opponents_hands_cbbc = opponents_hands_cbb
                final_my_hands_cbbc = my_hands_cbbc
                final_opponents_hands_bc = opponents_hands_b
                final_my_hands_bc = my_hands_bc
                final_opponents_hands_bbc = opponents_hands_bbc
                final_my_hands_bbc = my_hands_bb

                # String version for Equilab
                final_opponents_hands_cc_string = convert_hands_to_range(opponents_hands_c, flop)
                final_my_hands_cc_string = convert_hands_to_range(my_hands_cc, flop)
                final_opponents_hands_cbc_string = convert_hands_to_range(opponents_hands_cbc, flop)
                final_my_hands_cbc_string = convert_hands_to_range(my_hands_cb, flop)
                final_opponents_hands_cbbc_string = convert_hands_to_range(opponents_hands_cbb, flop)
                final_my_hands_cbbc_string = convert_hands_to_range(my_hands_cbbc, flop)
                final_opponents_hands_bc_string = convert_hands_to_range(opponents_hands_b, flop)
                final_my_hands_bc_string = convert_hands_to_range(my_hands_bc, flop)
                final_opponents_hands_bbc_string = convert_hands_to_range(opponents_hands_bbc, flop)
                final_my_hands_bbc_string = convert_hands_to_range(my_hands_bb, flop)







            else:
                # Hands version
                final_my_hands_cc = my_hands_c
                final_opponents_hands_cc = opponents_hands_cc
                final_my_hands_cbc = my_hands_cbc
                final_opponents_hands_cbc = opponents_hands_cb
                final_my_hands_cbbc = my_hands_cbb
                final_opponents_hands_cbbc = opponents_hands_cbbc
                final_my_hands_bc = my_hands_b
                final_opponents_hands_bc = opponents_hands_bc
                final_my_hands_bbc = my_hands_bbc
                final_opponents_hands_bbc = opponents_hands_bb

                # String version for Equilab
                final_my_hands_cc_string = convert_hands_to_range(my_hands_c, flop)
                final_opponents_hands_cc_string = convert_hands_to_range(opponents_hands_cc, flop)
                final_my_hands_cbc_string = convert_hands_to_range(my_hands_cbc, flop)
                final_opponents_hands_cbc_string = convert_hands_to_range(opponents_hands_cb, flop)
                final_my_hands_cbbc_string = convert_hands_to_range(my_hands_cbb, flop)
                final_opponents_hands_cbbc_string = convert_hands_to_range(opponents_hands_cbbc, flop)
                final_my_hands_bc_string = convert_hands_to_range(my_hands_b, flop)
                final_opponents_hands_bc_string = convert_hands_to_range(opponents_hands_bc, flop)
                final_my_hands_bbc_string = convert_hands_to_range(my_hands_bbc, flop)
                final_opponents_hands_bbc_string = convert_hands_to_range(opponents_hands_bb, flop)

            # Flop as string (won't get a card twice due to exceptions thrown far above)
            if board_type == "rainbow":
                final_flop_string = m2[flop[0]] + "s" + m2[flop[1]] + "h" + m2[flop[2]] + "c"
            elif board_type == "two-tone":
                final_flop_string = m2[flop[0]] + "s" + m2[flop[1]] + "h" + m2[flop[2]] + "s"
            elif board_type == "monotone":
                final_flop_string = m2[flop[0]] + "s" + m2[flop[1]] + "s" + m2[flop[2]] + "s"




              # #### 3) Equilab control mouse and save equity
            from time import sleep


            equity_cc = 0.50
            equity_cbc = 0.50
            equity_cbbc = 0.50
            equity_bc = 0.50
            equity_bbc = 0.50
            python_bin = "/Users/petermyers/Documents/pbots_calc-master/venv/bin/python"

            actions = ["cc", "cbc", "cbbc", "bc", "bbc"]
            mine_temp = [final_my_hands_cc_string, final_my_hands_cbc_string, final_my_hands_cbbc_string, final_my_hands_bc_string, final_my_hands_bbc_string]
            opponents_temp = [final_opponents_hands_cc_string, final_opponents_hands_cbc_string, final_opponents_hands_cbbc_string, final_opponents_hands_bc_string, final_opponents_hands_bbc_string]


            for action, my_hands_string, opponents_hands_string in zip(actions, mine_temp, opponents_temp):
                # If empty range, continue
                if len(my_hands_string) == 0 or len(opponents_hands_string) == 0:
                    continue



                # Equity calculation
                try:
                    command = "source /Users/petermyers/Documents/pbots_calc-master/venv/bin/activate; /Users/petermyers/Documents/pbots_calc-master/python/calculator.sh {}:{} {}".format(my_hands_string, opponents_hands_string, final_flop_string)
                    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
                    raw_equity = ast.literal_eval(process.communicate()[0].strip().decode("utf-8"))[0][1]
                except:
                    raw_equity = 0.50
                    print("shell command failed")



                # Adjust for position and implied odds
                # (assuming cat3 is only one with implied odds; not entirely true but fine)
                # 3 represents +/- 3%, 0.5 represents maximum difference expected
                # - 0.03 represents unrealized equity from being out of position
                positional_adjustment = 0.03 if my_position_ip else -0.03
                if action == "cc":
                    implied_odds_adjustment = max(-3,min(3, ((my_cat3_pct_cc - opponents_cat3_pct_cc)/0.5)*3))/100
                    equity_cc = raw_equity + positional_adjustment + implied_odds_adjustment
                elif action == "cbc":
                    implied_odds_adjustment = max(-3,min(3, ((my_cat3_pct_cbc - opponents_cat3_pct_cbc)/0.5)*3))/100
                    equity_cbc = raw_equity + positional_adjustment + implied_odds_adjustment
                elif action == "cbbc":
                    implied_odds_adjustment = max(-3,min(3, ((my_cat3_pct_cbbc - opponents_cat3_pct_cbbc)/0.5)*3))/100
                    equity_cbbc = raw_equity + positional_adjustment + implied_odds_adjustment
                elif action == "bc":
                    implied_odds_adjustment = max(-3,min(3, ((my_cat3_pct_bc - opponents_cat3_pct_bc)/0.5)*3))/100
                    equity_bc = raw_equity + positional_adjustment + implied_odds_adjustment
                else:
                    implied_odds_adjustment = max(-3,min(3, ((my_cat3_pct_bbc - opponents_cat3_pct_bbc)/0.5)*3))/100
                    equity_bbc = raw_equity + positional_adjustment + implied_odds_adjustment







            #### 4) The pot size in each situation including fold situations
            if my_position_ip:
                winnings_cc = pot_size*equity_cc
                winnings_cbf = pot_size + (pot_size*my_bet_size)
                cbc_pot_size = (pot_size + (pot_size*my_bet_size)*2)
                winnings_cbc = cbc_pot_size*equity_cbc
                winnings_cbbf = 0
                winnings_cbbc = (cbc_pot_size + (cbc_pot_size*opponents_bet_size)*2)*equity_cbbc
                winnings_bf = 0
                bc_pot_size = (pot_size + (pot_size*opponents_bet_size)*2)
                winnings_bc = bc_pot_size*equity_bc
                winnings_bbf = (bc_pot_size + (bc_pot_size*my_bet_size))
                winnings_bbc = (bc_pot_size + (bc_pot_size*my_bet_size)*2)*equity_bbc

                my_investment_cc = my_investment
                my_investment_cbf = my_investment + (my_bet_size*pot_size)
                my_investment_cbc = my_investment + (my_bet_size*pot_size)
                my_investment_cbbf = my_investment + (my_bet_size*pot_size)
                my_investment_cbbc = my_investment + (my_bet_size*pot_size) + (cbc_pot_size*opponents_bet_size)
                my_investment_bf = my_investment
                my_investment_bc = my_investment +  (pot_size*opponents_bet_size)
                my_investment_bbf = my_investment + (pot_size*opponents_bet_size) + (bc_pot_size*my_bet_size)
                my_investment_bbc = my_investment + (pot_size*opponents_bet_size) + (bc_pot_size*my_bet_size)



            else:
                winnings_cc = pot_size*equity_cc
                winnings_cbf = 0
                cbc_pot_size = (pot_size + (pot_size*opponents_bet_size)*2)
                winnings_cbc = cbc_pot_size*equity_cbc
                winnings_cbbf = (cbc_pot_size + (cbc_pot_size*my_bet_size))
                winnings_cbbc = (cbc_pot_size + (cbc_pot_size*my_bet_size)*2)*equity_cbbc
                winnings_bf = pot_size + (pot_size*my_bet_size)
                bc_pot_size = (pot_size + (pot_size*my_bet_size)*2)
                winnings_bc = bc_pot_size*equity_bc
                winnings_bbf = 0
                winnings_bbc = (bc_pot_size + (bc_pot_size*opponents_bet_size)*2)*equity_bbc

                my_investment_cc = my_investment
                my_investment_cbf = my_investment
                my_investment_cbc = my_investment + (opponents_bet_size*pot_size)
                my_investment_cbbf = my_investment + (opponents_bet_size*pot_size) + (cbc_pot_size*my_bet_size)
                my_investment_cbbc = my_investment + (opponents_bet_size*pot_size) + (cbc_pot_size*my_bet_size)
                my_investment_bf = my_investment + (pot_size*my_bet_size)
                my_investment_bc = my_investment + (pot_size*my_bet_size)
                my_investment_bbf = my_investment + (pot_size*my_bet_size)
                my_investment_bbc = my_investment + (pot_size*my_bet_size) + (bc_pot_size*opponents_bet_size)

            # Final profit amount
            profit = (winnings_cc-my_investment_cc)*chance_cc + \
                     (winnings_cbf-my_investment_cbf)*chance_cbf + \
                     (winnings_cbc-my_investment_cbc)*chance_cbc + \
                     (winnings_cbbf-my_investment_cbbf)*chance_cbbf + \
                     (winnings_cbbc-my_investment_cbbc)*chance_cbbc + \
                     (winnings_bf-my_investment_bf)*chance_bf + \
                     (winnings_bc-my_investment_bc)*chance_bc + \
                     (winnings_bbf-my_investment_bbf)*chance_bbf + \
                     (winnings_bbc-my_investment_bbc)*chance_bbc

            print("Profit: %.3f" % (profit))


            if profit > max_profit:
                max_profit = profit
                profits['0b'].append(cat3_level_0b)
                profits['1b'].append(cat3_level_1b)
                profits['2b'].append(cat3_level_2b)
                profits['profit'].append(profit)
                profits['flop'].append(flop)
                profits['board_type'].append(board_type)

            if profit < last_profit and profit_became_worse:
                profit_became_worse_twice = True
            elif profit < last_profit:
                profit_became_worse = True
            last_profit = profit

            return profit_became_worse_twice, profit_became_worse, profits, last_profit, max_profit








        # ***

        # Initialize loop
        profit_became_worse = False
        profit_became_worse_twice = False
        for cat3_level_0b in range(1, 5):
            cat3_level_1b = math.ceil(cat3_level_0b * 0.66)
            cat3_level_2b = math.ceil(cat3_level_1b * 0.66)

            # Run main
            profit_became_worse_twice, profit_became_worse, profits, last_profit, max_profit = \
                main(cat3_level_0b, cat3_level_1b, cat3_level_2b, max_profit, last_profit, profits, profit_became_worse, profit_became_worse_twice)

            if profit_became_worse: # Used to be profit_became_worse_twice
                break

        # Set best 0b
        max_profit_index = np.argmax(profits['profit'])
        cat3_level_0b = profits['0b'][max_profit_index]
        cat3_level_1b = profits['1b'][max_profit_index]
        cat3_level_2b = profits['2b'][max_profit_index]


        # Try 8, 10, 16

        for cat3_level_0b in [8, 10, 15, 16]:
            # if last_profit < max_profit and cat3_level_0b == 10: # Skip 10 if 8 gave no improvement (but still continue to 16 for now)
            #     continue
            cat3_level_1b = math.ceil(cat3_level_0b * 0.66)
            cat3_level_2b = math.ceil(cat3_level_1b * 0.66)

            # Run main
            profit_became_worse_twice, profit_became_worse, profits, last_profit, max_profit = \
                main(cat3_level_0b, cat3_level_1b, cat3_level_2b, max_profit, last_profit, profits, profit_became_worse, profit_became_worse_twice)

        # Set best 0b
        max_profit_index = np.argmax(profits['profit'])
        cat3_level_0b = profits['0b'][max_profit_index]
        cat3_level_1b = profits['1b'][max_profit_index]
        cat3_level_2b = profits['2b'][max_profit_index]




        # Commenting out because it improves profit by only 1% and increases computation time drastically, and makes it hard to memorize and record
        # Commenting out because it improves profit by only 1% and increases computation time drastically, and makes it hard to memorize and record
        # Commenting out because it improves profit by only 1% and increases computation time drastically, and makes it hard to memorize and record
        #
        #
        # profit_became_worse = False
        # profit_became_worse_twice = False
        # for new_cat3_level_1b in range(cat3_level_1b-1, 20):
        #     if new_cat3_level_1b == cat3_level_1b:
        #         continue
        #     if new_cat3_level_1b > cat3_level_0b + 1: # Current constraint on this being lte than about 0b, just to speed it up
        #         break
        #     cat3_level_1b = new_cat3_level_1b
        #     cat3_level_2b = math.ceil(cat3_level_1b * 0.66)
        #
        #     # Run main
        #     profit_became_worse_twice, profit_became_worse, profits, last_profit, max_profit = \
        #         main(cat3_level_0b, cat3_level_1b, cat3_level_2b, max_profit, last_profit, profits, profit_became_worse, profit_became_worse_twice)
        #
        #     if profit_became_worse: # Used to be profit_became_worse_twice
        #         break
        #
        # # Set best 1b
        # max_profit_index = np.argmax(profits['profit'])
        # cat3_level_0b = profits['0b'][max_profit_index]
        # cat3_level_1b = profits['1b'][max_profit_index]
        # cat3_level_2b = profits['2b'][max_profit_index]
        #
        #
        #
        # profit_became_worse = False
        # profit_became_worse_twice = False
        # for new_cat3_level_2b in range(cat3_level_2b-1, 20):
        #     if new_cat3_level_2b == cat3_level_2b:
        #         continue
        #     if new_cat3_level_2b > cat3_level_1b + 1: # Current constraint on this being lte than about 0b, just to speed it up
        #         break
        #     cat3_level_2b = new_cat3_level_2b
        #
        #     # Run main
        #     profit_became_worse_twice, profit_became_worse, profits, last_profit, max_profit = \
        #         main(cat3_level_0b, cat3_level_1b, cat3_level_2b, max_profit, last_profit, profits, profit_became_worse, profit_became_worse_twice)
        #
        #     if profit_became_worse: # Used to be profit_became_worse_twice
        #         break

        print(profits)
        df = pd.DataFrame(profits)
        print(df)
        df.to_csv("../../reports/{}/{}/{}.csv".format(range_name, board_type, str(i).zfill(3)))
        i += 1









        #### 6) Repeat with different cat3 configurations (0b and 2b may have slight correlation, 1b is independent)


        #### 7) Next steps: Consider trying another hand and later new positions (or even new bet size; wouldn't hurt to tweak a bit)
