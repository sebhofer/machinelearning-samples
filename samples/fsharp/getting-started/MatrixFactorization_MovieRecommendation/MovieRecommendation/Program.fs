open System
open System.IO

open Microsoft.ML
open Microsoft.ML.Trainers

open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.ML.Trainers.Recommender

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

    let testData = reader testDataLocation

    let dataProcessingPipeline =
        (mlContext.Transforms.Conversion.MapValueToKey("UserId","UserIdEncoded") |> downcastPipeline)
         .Append(mlContext.Transforms.Conversion.MapValueToKey("MovieId","MovieIdEncoded")) |> downcastPipeline

    let matrixFactorizationTrainer =
        let options = MatrixFactorizationTrainer.Arguments()
        options.NumIterations <- 20
        options.K <- 100
        mlContext.Recommendation().Trainers.MatrixFactorization("UserIdEncoded", "MovieIdEncoded", "Label")

    Console.WriteLine("=============== Training the model ===============");
    let transformer = dataProcessingPipeline.Fit(trainingData)
    let transformedTrainingData = transformer.Transform(trainingData)
    let trainedModel =
        matrixFactorizationTrainer.Fit(transformedTrainingData)

    Console.WriteLine("=============== Evaluating the model ===============")
    let transformedTestData = transformer.Transform(testData)
    let prediction = trainedModel.Transform(transformedTestData)
    let metrics = mlContext.Regression.Evaluate(prediction, label = "Label", score = "Score")
    Console.WriteLine("The model evaluation metrics rms: " + string(metrics.Rms))
    Console.WriteLine("=============== End of process, hit any key to finish ===============")
    Console.ReadLine()

    0 // return an integer exit code
