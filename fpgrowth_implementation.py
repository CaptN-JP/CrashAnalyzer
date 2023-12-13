# Implement FPgrowth along with the support and confidence and consider only the specified columns. All the columns are categorical.
# The support and confidence should be specified by the user.
# The output should be a list of rules with support and confidence.
# The output should be sorted by confidence in descending order.
# The output should be printed in the following format:
# Rule 1: {A} -> {B} (support = x%, confidence = y%)

# Path: fpgrowth.py
import pandas as pd
import pyfpgrowth
import math

# constants
ASSOCIATION_RULE_THRESHOLD = 0.8
SUPPORT_THRESHOLD_PERCENTAGE = 10
MAX_NUM_OF_FREQUENT_PATTERNS = 30
MIN_LENGTH_OF_PATTERN = 3
# columns_to_mine = ["Collision Type", "Weather", "Surface Condition", "Light", "Traffic Control"]
columns_to_mine = ["Injury Severity", "Agency Name", "ACRS Report Type", "Route Type", "Cross-Street Type"]
ASSOCIATION_NUM_ATTRIBUTES = 4
SUPPORT_NUM_ATTRIBUTES = 5


df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
df = df[columns_to_mine]
df = df.dropna()
df = df.values.tolist()

# Check that the association_rule_attributes and support_num_attributes are less than the length of the columns_to_mine
if ASSOCIATION_NUM_ATTRIBUTES > len(columns_to_mine):
    print("association_num_attributes is greater than the length of columns_to_mine")
    exit(1)
if SUPPORT_NUM_ATTRIBUTES > len(columns_to_mine):
    print("support_num_attributes is greater than the length of columns_to_mine")
    exit(1)
if MIN_LENGTH_OF_PATTERN > len(columns_to_mine):
    print("The minimum number of attributes in frequent pattern is \
          greater than the length of columns_to_mine")
    exit(1)

# Calculate support_threshold from SUPPORT_THRESHOLD_PERCENTAGE
support_threshold = math.floor(len(df) * (SUPPORT_THRESHOLD_PERCENTAGE / 100))

patterns = pyfpgrowth.find_frequent_patterns(df, support_threshold)
rules = pyfpgrowth.generate_association_rules(patterns, ASSOCIATION_RULE_THRESHOLD)

# Print top 30 frequent patterns
# top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:NUM_OF_FREQUENT_PATTERNS]
# print(f"Top {NUM_OF_FREQUENT_PATTERNS} Frequent Patterns:")

top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop Frequent Patterns:\n")
counter = 0
for pattern, support in top_patterns:
    if counter == MAX_NUM_OF_FREQUENT_PATTERNS:
        break
    if len(pattern) == MIN_LENGTH_OF_PATTERN:
        print(f"{pattern}: support = {support}")
        counter += 1
if counter < MAX_NUM_OF_FREQUENT_PATTERNS:
    print(f"\nOnly {counter} frequent patterns found with \n\tMIN_LENGTH_OF_PATTERN = {MIN_LENGTH_OF_PATTERN}\n\tMAX_NUM_OF_FREQUENT_PATTERNS = {MAX_NUM_OF_FREQUENT_PATTERNS}")
else:
    print(f"\nTop {MAX_NUM_OF_FREQUENT_PATTERNS} Frequent Patterns printed")

print("\n______________________________________________________\n")

# # Print association rules
# print("\nAssociation Rules:")
# for rule, (consequent, confidence) in rules.items():
#     print(f"{rule} -> {consequent}: confidence = {round(confidence, 2)}\n")

# Print association rules in decreasing order of confidence and decreasing order of number of attributes in rule
sorted_rules = sorted(rules.items(), key=lambda x: (len(x[0]), x[1][1]), reverse=True)
print("\nAssociation Rules:")
for rule, (consequent, confidence) in sorted_rules:
    print(f"{rule} -> {consequent}: confidence = {round(confidence, 2)}\n")
