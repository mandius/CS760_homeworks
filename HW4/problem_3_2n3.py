import string
alpha = 0.5

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

	print("Total Characters: ", total_characters)
	print("characters in this language: ", len(cdictlist[l].keys()))
	print("sum_of_counts: ", sum_of_counts)

	for i in char_list:
		if i in cdictlist[l]:
			p_dict[i] = (cdictlist[l][i] + alpha)/(sum_of_counts + alpha *total_characters)
		else:
			p_dict[i] = (alpha)/(sum_of_counts + alpha *total_characters)
	pdictlist.append(p_dict)

for k in pdictlist:
	print(k)
	
			 
