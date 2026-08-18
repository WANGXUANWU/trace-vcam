# Thin I/O wrapper around the unmodified CRAN fdapace::VCAM function.
# The statistical implementation remains inside fdapace; this file only maps
# the subject-ID exchange table to the package API and serializes its output.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 10L) {
  stop("expected input, curves, fitted, add_nknot, add_order, vc_nknot, vc_order, grid_size, time_domain, covariate_domains")
}
input_path <- args[[1L]]
curves_path <- args[[2L]]
fitted_path <- args[[3L]]
parse_ints <- function(value) as.integer(strsplit(value, ",", fixed = TRUE)[[1L]])
add_nknot <- parse_ints(args[[4L]])
add_order <- parse_ints(args[[5L]])
vc_nknot <- parse_ints(args[[6L]])
vc_order <- parse_ints(args[[7L]])
grid_size <- as.integer(args[[8L]])
time_domain <- as.numeric(strsplit(args[[9L]], ",", fixed = TRUE)[[1L]])
covariate_domain_tokens <- strsplit(args[[10L]], ";", fixed = TRUE)[[1L]]
covariate_domains <- t(vapply(
  covariate_domain_tokens,
  function(value) as.numeric(strsplit(value, ",", fixed = TRUE)[[1L]]),
  numeric(2L)
))

suppressPackageStartupMessages(library(fdapace))
dat <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
x_names <- grep("^x_", names(dat), value = TRUE)
subjects <- unique(dat$subject_id)
Lt <- vector("list", length(subjects))
Ly <- vector("list", length(subjects))
X <- matrix(NA_real_, nrow = length(subjects), ncol = length(x_names))
ordered_rows <- integer(0L)
for (i in seq_along(subjects)) {
  rows <- which(dat$subject_id == subjects[[i]])
  rows <- rows[order(dat$time[rows])]
  ordered_rows <- c(ordered_rows, rows)
  Lt[[i]] <- dat$time[rows]
  Ly[[i]] <- dat$response[rows]
  for (k in seq_along(x_names)) {
    values <- dat[[x_names[[k]]]][rows]
    if (max(values) - min(values) > 1e-10) {
      stop("fdapace::VCAM requires time-invariant covariates within subject")
    }
    X[i, k] <- values[[1L]]
  }
}
grid_x <- sapply(seq_along(x_names), function(k) {
  seq(covariate_domains[k, 1L], covariate_domains[k, 2L], length.out = grid_size)
})
if (is.null(dim(grid_x))) grid_x <- matrix(grid_x, ncol = 1L)
grid_t <- seq(time_domain[[1L]], time_domain[[2L]], length.out = grid_size)

# VCAM contains plotting calls. Direct them to a temporary device without
# altering any estimation code or defaults.
pdf(tempfile(fileext = ".pdf"))
on.exit(dev.off(), add = TRUE)
fit <- fdapace::VCAM(
  Lt, Ly, X,
  optnAdd = list(nKnot = add_nknot, order = add_order, grid = grid_x),
  optnVc = list(nKnot = vc_nknot, order = vc_order, grid = grid_t)
)

curve_rows <- data.frame(
  component = "baseline", domain = "time", grid = fit$gridT,
  value = fit$beta0Est, stringsAsFactors = FALSE
)
for (k in seq_along(x_names)) {
  curve_rows <- rbind(
    curve_rows,
    data.frame(component = paste0("beta_", k), domain = "time", grid = fit$gridT,
               value = fit$betaEst[, k], stringsAsFactors = FALSE),
    data.frame(component = paste0("phi_", k), domain = paste0("covariate_", k),
               grid = fit$gridX[, k], value = fit$phiEst[, k], stringsAsFactors = FALSE)
  )
}
write.csv(curve_rows, curves_path, row.names = FALSE, quote = TRUE)
fitted_values <- unlist(fit$LyHat, use.names = FALSE)
write.csv(
  data.frame(row_id = dat$row_id[ordered_rows], prediction = fitted_values),
  fitted_path, row.names = FALSE, quote = TRUE
)
