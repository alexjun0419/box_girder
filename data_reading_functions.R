# Add custom functions you make to this file

# When using functions from this file, add something like "source(here::here(functions.r))" to the top,
# it's similar to how library() works

wrangle_columns <- function(columns) {
  return(columns |>
    mutate(...1 = NULL,
           CID = as.factor(CID),
           RP = factor(RP, ordered = TRUE),
           GMID = str_replace(GMID, "GM(\\d{1})$", "GM0\\1") |>
             as.factor(),
           LDI = case_when(
             DI > 0 ~ log(DI),
             DI <= 0 ~ log(1e-06)
           ),
           DI = ifelse(DI < 0, 0, DI),
           DI = ifelse(DI > 1, 1, DI),
           DCR = Dmax / Du
           )
    )
}

wrangle_gm <- function(gm) {
  return(gm |>
    mutate(
      TID = as.factor(TID),
      RP = factor(RP, ordered = TRUE),
      RPID = factor(RPID, ordered = TRUE),
      EventID = as.factor(EventID),
      LPGV = log(PGV),
      LRS_T1 = log(RS_T1),
      GMID = str_replace(GMID, "GM(\\d{1})$", "GM0\\1") |>
        as.factor()
      )
    )
}

merge_columns_gm <- function(columns, gm) {
  return(left_join(columns, gm,
                   by=c(
                     'LID'  = 'LID',
                     'T1'   = 'Time',
                     'RP'   = 'RP',
                     'GMID' = 'GMID'
                     )
                   ) |>
           select(
             -NPiles,
             # MeanPGA to StDD595 were labeled as not important for now
             -c("MeanPGA", "MeanPGV", "MeanCAV", "MeanAI", "MeanD575",
                "MeanD595", "StDPGA", "StDPGV", "StDCAV", "StDAI",
                "StDD575", "StDD595"),
             -RunTIme,
             # RS_T0 to RS_T10 were labeled as not important for now
             -c("RS_T0", "RS_T0p01", "RS_T0p02", "RS_T0p03", "RS_T0p05",
                "RS_T0p075", "RS_T0p1", "RS_T0p15", "RS_T0p2", "RS_T0p25",
                "RS_T0p3", "RS_T0p4", "RS_T0p5", "RS_T0p75",                  #"RS_T1",
                "RS_T1p5", "RS_T2", "RS_T3", "RS_T4", "RS_T5",
                "RS_T7p5", "RS_T10")
           )
         )
}

read_and_merge_files <- function(type) {
  if (type == "columns") {
    directory <- here::here("Data/ColumnsData")
  } else if (type == "gm") {
    directory <- here::here("Data/GMData")
  } else {
    stop("No selection for read_and_merge_files")
  }
  
  # List all files in the chosen directory with the "given".txt" filetype
  files <- list.files(path = directory, pattern = "txt", full.names = TRUE)
  
  # Initialize an empty list to store datasets
  datasets <- list()
  
  # Loop through each file, read it, and add LID
  for (file in files) {
    data <- read_csv(file)
    data$LID <- str_extract(basename(file), "\\d+")
    datasets <- append(datasets, list(data))
  }
  
  # Bind all the datasets into one dataframe
  dataset <- bind_rows(datasets)
  
  return(dataset)
}
