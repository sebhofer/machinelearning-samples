// Learn more about F# at http://fsharp.org

open System
open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.ML.Trainers.Recommender
open Microsoft.ML.Data
open System.IO

[<CLIMutable>]
type Movie =
    {
        MovieId : int
        MovieTitle: string
    }

[<CLIMutable>]
type MovieRating =
    {
        [<LoadColumn(0)>]
        UserId : double
        [<LoadColumn(1)>]
        MovieId: double
        [<LoadColumn(2)>]
        Label : single
    }

[<CLIMutable>]
type MovieRatingPrediction =
    {
        Label : single
        Score : single
    }

// Using the ml-latest-small.zip as dataset from https://grouplens.org/datasets/movielens/.
let modelsLocation = @"../../../../MLModels"

let datasetsLocation = @"../Data"
let trainingDataLocation = sprintf @"%s/recommendation-ratings-train.csv" datasetsLocation
let testDataLocation = sprintf @"%s/recommendation-ratings-test.csv" datasetsLocation
let moviesDataLocation = sprintf @"%s/movies.csv" datasetsLocation

/// Cast ML.NET pipeline object to IEstimator<ITransformer> interface
let downcastPipeline (pipeline : IEstimator<'a>) =
    match pipeline with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The pipeline has to be an instance of IEstimator<ITransformer>."

[<EntryPoint>]
let main argv =

    // Create MLContext to be shared across the model creation workflow objects 
    // Set a random seed for repeatable/deterministic results across multiple trainings.
    let mlContext = MLContext(seed = Nullable 0)

    let reader location = 
        mlContext.Data.ReadFromTextFile<MovieRating>(location, hasHeader = true, separatorChar = ',')

    let textLoader = 
       mlContext.Data.CreateTextReader(
           columns = 
               [|
                   TextLoader.Column("MovieId", Nullable DataKind.R8, 2)
                   TextLoader.Column("MovieTitle", Nullable DataKind.R8, 3)
               |],
           hasHeader = true, 
           separatorChar = ','
       )

    let trainingData = reader trainingDataLocation
    printfn "data %O" (trainingData.Preview(5))

    let testData = reader testDataLocation

    let dataProcessingPipeline = 
        (mlContext.Transforms.Conversion.MapValueToKey("UserId","UserIdEncoded") |> downcastPipeline)
         .Append(mlContext.Transforms.Conversion.MapValueToKey("MovieId","MovieIdEncoded")) |> downcastPipeline

    let matrixFactorizationTrainer =
        mlContext.Recommendation().Trainers.MatrixFactorization("UserIdEncoded", "MovieIdEncoded", "Label")

    let transformer = dataProcessingPipeline.Fit(trainingData)
    let transformedTrainingData = transformer.Transform(trainingData)
    let transformedTestData = transformer.Transform(testData)

    printfn "data %O" (transformedTrainingData.Preview(5))

    let trainedModel =
        matrixFactorizationTrainer.Fit(transformedTrainingData)

    let predictions = trainedModel.Transform(transformedTestData)
    for col in predictions.Schema do
        printfn "col %O" col

    let metrics = mlContext.Regression.Evaluate(predictions, label = "Label", score = "Score")

    printfn "metrics: %O" metrics.Rms


    0 // return an integer exit code
