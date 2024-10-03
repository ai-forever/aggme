from numpy import array  # NOQA

with open("inputs/bbox_drop.txt", "r") as f:
    input_bbox_drop = eval(f.read())

with open("inputs/bbox_hard.txt", "r") as f:
    input_bbox_hard = eval(f.read())

with open("inputs/bbox_soft.txt", "r") as f:
    input_bbox_soft = eval(f.read())

with open("outputs/bbox_drop.txt", "r") as f:
    output_bbox_drop = eval(f.read())

with open("outputs/bbox_hard.txt", "r") as f:
    output_bbox_hard = eval(f.read())

with open("outputs/bbox_soft.txt", "r") as f:
    output_bbox_soft = eval(f.read())

with open("inputs/mask_drop.txt", "r") as f:
    input_mask_drop = eval(f.read())

with open("inputs/mask_hard.txt", "r") as f:
    input_mask_hard = eval(f.read())

with open("inputs/mask_soft.txt", "r") as f:
    input_mask_soft = eval(f.read())

with open("outputs/mask_soft.txt", "r") as f:
    output_mask_soft = eval(f.read())

with open("outputs/mask_hard.txt", "r") as f:
    output_mask_hard = eval(f.read())

with open("outputs/mask_drop.txt", "r") as f:
    output_mask_drop = eval(f.read())
