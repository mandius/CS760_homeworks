import string
alpha = 0.5

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

x= []

for c in char_list:
	x.append(bag_of_words[c])

print(bag_of_words)
print(x)
	
