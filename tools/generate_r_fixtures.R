#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 1) {
  stop("usage: tools/generate_r_fixtures.R [manifest.json]")
}

if (!requireNamespace("flacco", quietly = TRUE)) {
  stop("R package 'flacco' is required")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("R package 'jsonlite' is required")
}

manifest_path <- if (length(args) == 1) args[[1]] else "tests/fixtures/r_flacco/manifest.json"
manifest_path <- normalizePath(manifest_path, mustWork = TRUE)
fixture_root <- dirname(manifest_path)
output_root <- file.path(fixture_root, "expected")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
feature_sets <- unlist(manifest$feature_sets, use.names = FALSE)

for (case in manifest$cases) {
  input_path <- file.path(fixture_root, case$file)
  input <- read.csv(input_path, check.names = FALSE)
  x_columns <- unlist(case$x_columns, use.names = FALSE)
  lower <- as.numeric(unlist(case$lower, use.names = FALSE))
  upper <- as.numeric(unlist(case$upper, use.names = FALSE))
  blocks <- as.integer(unlist(case$blocks, use.names = FALSE))
  set.seed(as.integer(case$seed))

  feature_object <- flacco::createFeatureObject(
    X = input[x_columns],
    y = input$y,
    lower = lower,
    upper = upper,
    blocks = blocks
  )

  values <- list()
  for (feature_set in feature_sets) {
    calculated <- flacco::calculateFeatureSet(feature_object, set = feature_set)
    calculated[[paste0(feature_set, ".costs_runtime")]] <- NULL
    calculated[[paste0(feature_set, ".costs_fun_evals")]] <- NULL
    values <- c(values, calculated)
  }
  values <- lapply(values, function(value) {
    if (length(value) != 1) {
      stop("fixture outputs must be scalar")
    }
    unname(value)
  })

  output <- list(
    schema_version = manifest$schema_version,
    case = case$name,
    input_file = case$file,
    input_sha256 = case$sha256,
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    flacco_version = as.character(utils::packageVersion("flacco")),
    feature_sets = feature_sets,
    values = values
  )
  output_path <- file.path(output_root, paste0(case$name, ".json"))
  jsonlite::write_json(
    output,
    output_path,
    auto_unbox = TRUE,
    digits = 17,
    pretty = TRUE,
    na = "string"
  )
  cat("wrote", output_path, "\n")
}
