import tkinter as tk
import TwitterScrapping as ts
import joblib


root = tk.Tk()
bg = "#B6CFC2"
root.geometry("500x250")
root.title("Personality Prediction")
root.configure(bg = bg)
model = joblib.load("mbti-model-nb.sav")
vect = joblib.load(open("vectorizer.pickle","rb"))

def getResult(userName):
    user_tweets = ts.scrape(userName)
    user_vect = vect.transform([user_tweets])
    user_type = model.predict(user_vect)
    result = tk.Label(root,text = user_type[0],bg = bg)
    result.place(x = 200,y = 120)



user_name = tk.Label(root,text = "Twitter User Handle : ",bg=bg)
user_entry = tk.Entry(root,width = 20,justify = tk.CENTER)
user_name.place(x = 20, y = 40) 
user_entry.place(x = 180, y = 40)

find_button = tk.Button(root,text = "Find Personality",bg = bg,command = lambda:getResult(user_entry.get()))
find_button.place(x = 180 , y = 80)



root.mainloop()
