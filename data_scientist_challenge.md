# Qogita's Data Science Technical Challenge

Congratulations on reaching the technical challenge stage of the interview process at Qogita.

We are excited to potentially have you join the Science team, where you will work alongside brilliant people to build a revolutionary global wholesale B2B platform.

## Contacting us

If you run into technical problems completing the exercises, please do get in touch and let us know so that we can help. You can contact the Data Science team by email at `team-science@qogita.com`.

## Overview

The exercise is broken down in to five parts. You will need to complete all tasks marked as mandatory, and we would like to see you complete or attempt the optional tasks.

1. Design, fit, evaluate and store a model, based on the **API data** made available (mandatory)
2. Build a FastAPI web application to serve, via REST requests, the predictions made by the model (strongly encouraged, but optional)
3. Write a Docker file that you would use to deploy the FastAPI application to a container (optional)
4. Submit your solution using a **private** Github repository! (mandatory)
5. Produce a presentation discussing your solution (max 15 min) (mandatory)

This challenge should take you at maximum a working day to complete, and you will be given a week to do this.

## Data

You can find the API documentation here [https://qogita.notion.site/Qogita-Buyer-API-Beta-556b22869bcb47d2bffac8d2a8c7076a](https://www.notion.so/556b22869bcb47d2bffac8d2a8c7076a?pvs=21)

To select your data sample:

- Use the product search endpoint
- Search all in-stock products in all categories with stock_availability=in_stock, page=10, and size=250
    - do not filter by keyword, brand, category, has_deals, or cart_allocation_qid

Notes: 

- you will not need a bearer token if you are searching only on `stock_availability`, page and size
- don't worry about price, quantity, units being empty — these are for authenticated users 🙂

## Guidance

The modelling task has been left open for you to be as creative as possible; we appreciate that data scientists come from a broad range of backgrounds and bring with them a highly diverse set of skills. We have chosen to leave the modelling aspect of the technical exercise open to you, so that you can show us what you're best at.

### 1. Modelling

In this technical challenge we would like you to perform a classification task using product data from our website.

To complete the challenge you will need to retrieve a sample of products from our catalog using our public product data API. We would like to see you visualise the sample to highlight important attributes of the data set.

In the sample you will have access to each product's **title**, **brand**, **image** and category label.

Using any combination of  **title**, **brand**, and **image** you should create a classification algorithm to predict the category label of the product. 

Use the category labels of the products to evaluate the accuracy of your classification algorithm.

Present your experiment with a writeup including sections introduction, aim, glossary of terms, data definition, method, results and conclusions. 

Summarize your results and conclusions, and offer comment on the strengths and weaknesses of the approach you have chosen. If you believe there are better alternatives, explain what these are and why they would be an improvement.

### 2. Build a FastAPI web application

We would like you to build a FastAPI REST application that can serve the classifications for the model you have developed in step 1.

The web application should accept a request in the form of a JSON serialised request and return the output of your model as a JSON serialised response, using the serialised model you created in step 1 to generate the output.

### Example:

In step 1 you built a model that classifies a product’s category. We would like you to provide an endpoint `http://localhost:5000/classify` for a request with the body:

```json
{
  "product":  "0123456789123",
  "feature_1": ...,
  "feature_2": ...,
  ...
}
```

which returns the predicted classification as a JSON response:

```json
{
  "product":  "0123456789123",
  "prediction": ...
}
```

To help us use the application, you should include a README.md file with an overview of the API route you have designed and how to make a request for data (e.g. an example of a JSON request body).

If you are not familiar with FastAPI it may be useful to refer to the guide https://fastapi.tiangolo.com/tutorial/first-steps/.

### 3. Create a docker file

The final coding element of the exercise is to write a Docker file that could be used to deploy and run the FastAPI application within a container.

The docker container should run the web application on startup and make it available for requests on port 5000.

### 4. Submit your solution

We would like you to commit your solution to a **private** Github repository and under `Settings` -> `Manage access` -> `Add people` invite the user `qogita-interview-bot`. Also please send an email to `team-science@qogita.com` to let us know that you have submitted your solution.

### 5. Presentation

You should prepare a short presentation (slides optional, maximum 15 minutes) on your solution. This is your opportunity to sell your work to the team. You might want to consider telling us about:

- The business context of the problem your model addresses
- The model you've designed and its performance
- How you could improve the model or approach
- Deployment and Maintenance