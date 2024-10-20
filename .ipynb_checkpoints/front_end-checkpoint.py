{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "89549b9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import numpy as np\n",
    "import pickle\n",
    "import pandas as pd\n",
    "#from flasgger import Swagger\n",
    "import streamlit as st "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2022b4b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle_in = open(\"model_trn.pkl\",\"rb\")\n",
    "classifier=pickle.load(pickle_in)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "792f463e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def welcome():\n",
    "    return \"Welcome All\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "769a534c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_approval(input_row):\n",
    "    \n",
    "    prediction=classifier.predict([input_row])\n",
    "    print(prediction)\n",
    "    return prediction\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "475f1e24",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-29 15:56:19.984 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:19.987 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:19.990 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:19.992 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:19.995 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:19.997 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:19.998 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.000 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.002 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.004 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.007 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.009 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.011 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.012 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.014 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.015 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.016 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.018 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.020 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.022 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.023 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.025 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.026 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.028 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.030 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.032 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.036 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.038 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.041 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.044 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.047 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.052 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-29 15:56:20.055 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    st.title(\"Bank Authenticator\")\n",
    "    html_temp = \"\"\"\n",
    "    <div style=\"background-color:tomato;padding:10px\">\n",
    "    <h2 style=\"color:white;text-align:center;\">Streamlit Bank Authenticator ML App </h2>\n",
    "    </div>\n",
    "    \"\"\"\n",
    "    st.markdown(html_temp,unsafe_allow_html=True)\n",
    "    variance = st.text_input(\"Variance\",\"Type Here\")\n",
    "    skewness = st.text_input(\"skewness\",\"Type Here\")\n",
    "    curtosis = st.text_input(\"curtosis\",\"Type Here\")\n",
    "    entropy = st.text_input(\"entropy\",\"Type Here\")\n",
    "    \n",
    "    result=\"\"\n",
    "    \n",
    "    input_row= [variance,skewness,curtosis, entropy]\n",
    "    \n",
    "    if st.button(\"Predict\"):\n",
    "        result=predict_approval(input_row)\n",
    "    st.success('The output is {}'.format(result))\n",
    "    if st.button(\"About\"):\n",
    "        st.text(\"Lets LEarn\")\n",
    "        st.text(\"Built with Streamlit\")\n",
    "\n",
    "if __name__=='__main__':\n",
    "    main()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3acfd17b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
