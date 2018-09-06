import numpy as np
import linecache
#from nltk.stem import WordNetLemmatizer

#lemmatizer = WordNetLemmatizer()
fname = 'data/advising_only/advising_lemmatized_train.txt'
target = np.load('train_data/train_advising/train_target.npy')

while True:

    command = input("Do you want an example? : [y] or [n] =================================================================")

    #idx = int(input("What number of example do you want to see? :  index starts from 0 : "))

    

    if command == 'n':
        break
    if command == 'y':
        idx = int(input("What number of example do you want to see? : index starts from 0 : "))
        print('\n\nIn '+str(idx)+'th sample:')
        print("\nQuestion : ---------------------------------------------------------------------------------------------\n")
        question = linecache.getline(fname,idx*101+1)
        #import pdb; pdb.set_trace()
        question = question.split()
        for word in question:
            #word2 = lemmatizer.lemmatize(word) # lemmatizer
            if word != '__eot__':
                print( word , end=' ')
                #print( word2, end=' ')
            else:
                print( word+'\n')
                #print( word2+'\n')
        
        print("\n\nAnswer : ---------------------------------------------------------------------------------------------\n")
        print(str(target[idx])+':', linecache.getline(fname, idx*101+2+target[idx]))

        print("\n\nAnswer candidates : ----------------------------------------------------------------------------------\n")
        for i in range(100):
            answer = linecache.getline(fname, idx*101+2+i)
            #answer2 = lemmatizer.lemmatize(answer)
            if i == target[idx]:
                save_answer = answer
            
            print(str(i)+':',answer, end='')
            #print(i, ' : ',answer2)

        # print("\n\nAnswer : ---------------------------------------------------------------------------------------------\n")
        # print(target[idx], save_answer,'\n')
        #print(target[target_idx], lemmatizer.lemmatize(save_answer),'\n')
        #target_idx += 1

    else:
        continue

#f.close()
