# I/O-only wrapper for the pinned, unmodified VCAMLasso author source.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) {
  stop("expected vendor source, input, result, mse, d, df, seed")
}
vendor_source <- args[[1L]]
input_path <- args[[2L]]
result_path <- args[[3L]]
mse_path <- args[[4L]]
d <- as.integer(args[[5L]])
df <- as.integer(strsplit(args[[6L]], ",", fixed = TRUE)[[1L]])
seed <- as.integer(as.numeric(args[[7L]]) %% 2147483647)
if (length(df) != 1L + 2L * d) stop("df must have length 1 + 2*d")

suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(splines2))
source(vendor_source, local = environment(), encoding = "UTF-8")
dat <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
x_names <- grep("^x_", names(dat), value = TRUE)
if (length(x_names) != d) stop("covariate width does not match d")
set.seed(seed)
fit <- VCAMLasso(
  X = as.matrix(dat[, x_names, drop = FALSE]),
  TIME = dat$time,
  Y = dat$response,
  d = d,
  df = df
)
result <- cbind(row_id = dat$row_id, fit$result.data)
write.csv(result, result_path, row.names = FALSE, quote = TRUE)
writeLines(format(fit$MSE, digits = 17L, scientific = TRUE), mse_path)
