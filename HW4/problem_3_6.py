import string
import math
alpha = 0.5


def print_exp(log_prob):
	int_part = math.floor(log_prob)
	frac_part = log_prob - int_part 

	exp_mantissa = math.exp(frac_part)
	exp_exp =  int_part
	
	return (str(exp_mantissa) +"e"+ str(exp_exp))

#Loop for parsing the files:
cdictlist =  [] # Order ["e", "s", "j"]
for l in ["e","s","j"]:
	c_dict = {}
	for i in range(0,10):
		filename = "languageID/"+ l +str(i)+".txt"
		fh = open(filename, "r")
		for line in fh:
			for c in line:
				if not c == "\n":
					if c in c_dict:
						c_dict[c] = c_dict[c] +1
					else:
						c_dict[c] =1
	cdictlist.append(c_dict)


#Create a list of characters we want to search
char_list = list(string.ascii_lowercase)
char_list.append(" ")



#Loop for Calculating class conditional probabilities:
pdictlist = []
for l in range(0,3):
	p_dict = {}
	total_characters  = len(char_list)
	sum_of_counts = 0
	for i in cdictlist[l].keys():
		sum_of_counts = sum_of_counts + cdictlist[l][i]

	#print("Total Characters: ", total_characters)
	#print("characters in this language: ", len(cdictlist[l].keys()))
	#print("sum_of_counts: ", sum_of_counts)

	for i in char_list:
		if i in cdictlist[l]:
			p_dict[i] = (cdictlist[l][i] + alpha)/(sum_of_counts + alpha *total_characters)
		else:
			p_dict[i] = (alpha)/(sum_of_counts + alpha *total_characters)
	pdictlist.append(p_dict)


filename =  "languageID/e10.txt"

fh = open(filename,  "r")

char_list =  list(string.ascii_lowercase)
char_list.append(" ")

bag_of_words = {}

for c in char_list:
	bag_of_words[c] = 0

for line in fh:
	for c in line:
		if not c == "\n":
			bag_of_words[c] =  bag_of_words[c]+1

#Printing Likelihood
#for english:
log_probability = 0
for c in char_list:
	log_probability = log_probability + bag_of_words[c] * math.log(pdictlist[0][c])


log_prior = math.log(1/3)
log_probability = log_probability + log_prior


print("For English:")
print(log_probability, print_exp(log_probability))

#for spanish:
log_probability = 0
for c in char_list:
	log_probability = log_probability + bag_of_words[c] * math.log(pdictlist[1][c])

log_prior = math.log(1/3)
log_probability = log_probability + log_prior

print("For Spanish:")
print(log_probability, print_exp(log_probability))

#for japanese:
log_probability = 0
for c in char_list:
	log_probability = log_probability + bag_of_words[c] * math.log(pdictlist[2][c])

log_prior = math.log(1/3)
log_probability = log_probability + log_prior

print("For Japanese:")
print(log_probability, print_exp(log_probability))
	


