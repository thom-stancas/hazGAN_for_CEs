# remember to select source on save
.libPaths(c("~/R/libs/hazGAN", .libPaths()))
library(zoo)
library(eva)
library(extRemes)
library(dplyr)
library(lubridate)
library(future)
library(furrr)
library(data.table)
library(progress)  # Add this
library(magrittr)  # Optional but recommended for pipe operator
library(goftest)

########### HELPER FUNCTIONS ###################################################
`%ni%` <- Negate(`%in%`)



res2str <- function(res){
    string <- paste0(res[1], "x", res[2])
    return(string)
}


standardise_by_month <- function(df, var) {
    df$month <- months(df$time)
    df <- df[,c(var, "month", "grid")]
    monthly_median <- aggregate(. ~ month + grid, df, median)
    df$monthly_median <- left_join(
        df[, c("month", "grid")],
        monthly_median,
        by = c("month" = "month", "grid" = "grid")
    )[[var]]
    df[[var]] <- df[[var]] - df$monthly_median
    return(df[[var]])
}



monthly_medians <- function(df, var) {
    df <- df[, c(var, "time", "grid")]
    df$month <- months(df$time)
    monthly_median <- aggregate(. ~ month + grid,
                                df[, c(var, "grid", "month")],
                                median)
    return(monthly_median)
}



ecdf <- function(x) {
    #' Empirical cumulative distribution function (ECDF)
    #'
    #' Create an empirical CDF function from a numeric vector. The returned
    #' object behaves like the base::ecdf result: it is a step function that
    #' maps values to their empirical cumulative probability.
    #' 
    #' For a given value x, the empirical CDF is defined as:
    #' rank-like cumulative probability = count(training values <= x) / (n + 1)
    #'
    #' @param x Numeric vector of observations.
    #' @return A function (of class c("ecdf", "stepfun")) that when called with a
    #'   numeric vector returns the empirical cumulative probabilities.
    #' @examples
    #' x <- c(1,2,2,3)
    #' F <- ecdf(x)
    #' F(2)  # returns 0.6, since 3 out of 4 values are <= 2


    # Sort the training values and get length
    x <- sort(x)
    n <- length(x)

    # If there are no values, throw an error
    if (n < 1) {
        stop("'x' must have 1 or more non-missing values")
    }

    # Create the empirical CDF function using approxfun
    # Compute the ranks and cumulative probabilities
    vals <- unique(x)
    rval <- approxfun(vals, cumsum(tabulate(match(x, vals))) / (n + 1),
                    method = "constant", #yleft = 0, yright = 1,
                    rule = 2, # take values at extremes
                    f = 0, ties = "ordered")
    class(rval) <- c("ecdf", "stepfun", class(rval))
    assign("nobs", n, envir = environment(rval))
    attr(rval, "call") <- sys.call()
    rval
}



scdf <- function(train, loc, scale, shape, cdf = pgpd){
    #' Semi-parametric CDF function for exceedances above a threshold.
    #' Creates a calculator which computes the semi-parametric CDF for a given 
    #' set of values based on a fitted GPD model.
    #' 
    # Note, trialing using excesses and setting loc=0
    # This is for flexibility with cdf choice
    calculator <- function(x){
        u <- ecdf(train)(x) # First compute the empirical CDF for all values in x
        pthresh <- ecdf(train)(loc) # And the empirical CDF at the threshold
        tail_mask <- x > loc # Identify values in x that exceed the threshold
        x_tail <- x[tail_mask] 
        exceedances <- x_tail - loc
        u_tail <- 1 - (1 - pthresh) * (1 - cdf(exceedances, scale=scale, shape=shape)) 
        # Compute the semi-parametric CDF for exceedances
        u[tail_mask] <- u_tail
        return(u)
    }
    return(calculator)
}

hurdle_ecdf <- function(train) {
    train <- train[is.finite(train)]
    pos_train <- train[train > 0]

    function(x) {
        u <- rep(NA_real_, length(x))

        zero_mask <- x == 0
        pos_mask    <- x > 0

        u[zero_mask] <- 0

        if (length(pos_train) > 0) {
            Fpos <- ecdf(pos_train)
            u[pos_mask] <- 0.5 + 0.5 * Fpos(x[pos_mask])
        } else {
            u[pos_mask] <- 1
        }

        u[x < 0] <- NA_real_
        u
    }
}

progress_bar <- function(n, prefix = "", suffix = "") {
    pb <- utils::txtProgressBar(min = 0, max = n, style = 3)
    function(i) {
        utils::setTxtProgressBar(pb, i)
        if (i == n) close(pb)
    }
}

########### EVT FUNCTIONS ######################################################
gridsearch <- function(series, var, qmin = 60, qmax = 99, rmax = 14) {
    "Unit tests for this?"
    qvec <- c(qmin:qmax) / 100
    rvec <- c(rmin:rmax)

    print("Initial data summary:")
    print(summary(series[[var]]))

    nclusters <- matrix(nrow = length(rvec), ncol = length(qvec))
    ext_ind   <- matrix(nrow = length(rvec), ncol = length(qvec))
    pvals     <- matrix(nrow = length(rvec), ncol = length(qvec))

    series_var <- series[[var]]
    thresholds  <- quantile(series_var, qvec)

    print("Testing combinations:")
    for (i in seq_along(rvec)){
        for (j in seq_along(qvec)){
            thresh <- thresholds[j]

            d <- decluster(series_var, thresh = thresh,
                     r = rvec[i], method = "runs")

            # NOTE: theta = 1 a lot, double-check?
            e <- extremalindex(c(d), thresh, r = rvec[i],
                         method = "runs") # Coles (2001) §5.3.2

            # print(sprintf("r=%d, q=%.2f: ext_ind=%.3f",
            #               rvec[i], qvec[j], e[["extremal.index"]]))

            p <- Box.test(c(d)[c(d) > thresh], type = "Ljung")

            nclusters[i, j] <- e[["number.of.clusters"]]
            ext_ind[i, j]   <- e[["extremal.index"]]
            pvals[i, j]     <- p$p.value
        }
    }
    print("Before filtering:")
    print(table(is.finite(nclusters)))

    print("After extremal index filter:")
    nclusters[ext_ind < 0.8] <- -Inf # theta < 1 => extremal dependence
    print(table(is.finite(nclusters)))
    print("After p-value filter:")

    nclusters[pvals < 0.1]  <- -Inf  # H0: independent exceedances
    print(table(is.finite(nclusters)))

    ind <- which(nclusters == max(nclusters), arr.ind = TRUE)
    r <- rvec[ind[1]]
    q <- qvec[ind[2]]
    p <- pvals[ind[1], ind[2]]
    return(list(r = r, q = q, p = p))
}
event_extractor <- function(daily, var, rfunc) {
    series <- aggregate(. ~ time, daily[, c("time", var)], rfunc)

    # gridsearch run lengths and thresholds
    result <- gridsearch(series, var)
    r <- result$r
    q <- result$q
    p <- result$p

    thresh <- quantile(series[[var]], q)
    cat(paste0(
        "Final selection from gridsearch:\n",
        "Run length: ", r, "\n",
        "Quantile: : ", q, "\n",
        "Threshold: ", round(thresh, 4), "\n",
        "P-value (H0:independent): ", round(p, 4), "\n"
    ))

    # final declustering
    declustering <- decluster(series[[var]], thresh, r = r)
    events <- attr(declustering, "clusters")
    times <- series$time[series[[var]] > thresh]
    variable <- series[[var]][series[[var]] > thresh]
    metadata <- data.frame(time = times, event = events, variable = variable)

    # event stats
    events <- metadata %>%
        group_by(event) %>%
        mutate(event.size = n()) %>%
        slice(which.max(variable)) %>%
        summarise(
            variable = max(variable),
            time = time,
            event.size = event.size
        )

    # Ljung-box again
    p <- Box.test(c(events$variable), type = "Ljung")$p.value
    cat(paste0("Final Ljung-Box p-value: ", round(p, 4), '\n'))

    # event frequency
    m <- nrow(events)
    nyears <- length(unique(year(daily$time)))
    lambda <- m / nyears
    metadata$lambda <- lambda
    cat(paste0("Number of events: ", m, '\n'))

    # assign return periods
    survival_prob <- 1 - (
        rank(events$variable, ties.method = "average") / (m + 1)
    )
    rp <- 1 / (lambda * survival_prob)
    events$event.rp <- rp

    # remaining metadata
    metadata <- left_join(metadata,
                        events[c("event", "event.rp", "event.size")],
                        by = c("event"))
    metadata <- metadata %>% rename_with(~ var, variable)

    return(metadata)
}

########## TRANSFORMS ##########################################################
gpdBackup <- function(var, threshold) {
    #' Fit a Generalized Pareto Distribution (GPD) to exceedances above a threshold using the POT package.
    #' 
    #' @param var Numeric vector of data to fit.
    #' @param threshold Numeric value representing the threshold above which to fit the GPD.
    #' 

    library(POT)
    # Define the Anderson-Darling test for goodness-of-fit
    # This creates a local function that:
    # 1. Builds a CDF function using pgpd.
    # 2. Runs ad.test(x, cdf).
    # 3. Returns the p-value.
    ad_test <- function(x, shape, scale, eps=0.05){
        cdf <- function(x) pgpd(x, loc = 0, shape = shape, scale = scale)
        result <- ad.test(x, cdf)
        return(list(p.value = result$p.value))
    }
    
    # Compute exceedances
    exceedances <- var[var > threshold] - threshold
    exceedances <- sort(exceedances)
    num.above <- length(exceedances)

    # Fit GPD using the POT package's fitgpd function
    fit <- fitgpd(var, threshold = threshold, est = "pwmu")
    scale <- fit$fitted.values[1]
    shape <- fit$fitted.values[2]

    # goodness-of-fit test
    gof  <- ad_test(exceedances, threshold, scale, shape)
  
    # Return the fitted parameters and goodness-of-fit statistics
    return(list(thresh=threshold, shape=shape, scale=scale,
              p.value=gof$p.value,
              num.above = num.above))
}


gpdBackupSeqTests <- function(var, thresholds) {
    #' Sequential goodness-of-fit tests for GPD fitting with fallback to POT package.
    #' This function attempts to fit a GPD to exceedances above a sequence of thresholds
    #' and performs goodness-of-fit tests. If the primary fitting method fails, it falls back to using the POT package.
    #' The forward stopping criterion is applied to determine the lowest threshold that passes the 
    #' goodness-of-fit test.
    
    # Initialize vectors to store results for each threshold
    nthresh <- length(thresholds)
    shapes <- vector(length=nthresh)
    scales <- vector(length=nthresh)
    p.values <- vector(length=nthresh)
    num.above <- vector(length=nthresh)

    # Loop over each threshold, fit GPD, and perform goodness-of-fit test
    for (k in seq_along(thresholds)) {
        thresh        <- thresholds[k]
        fit           <- gpdBackup(var, thresh) # Apply the backup GPD fitting function
        shapes[k]     <- fit$shape
        scales[k]     <- fit$scale
        p.values[k]   <- fit$p.value
        num.above[k]  <- fit$num.above
    }
  
    # ForwardStop <-  cumsum(-log(1 - p.values)) / (seq_along(p.values))
    # i.e. Stop when the cumulative sum of -log(1-p) divided by the index exceeds alpha
    ForwardStop <- rev(eva:::pSeqStop(rev(p.values))$ForwardStop)

    # Return results as a data frame
    out <- list(threshold = thresholds, num.above = num.above, p.value = p.values, 
              ForwardStop = ForwardStop, est.scale = scales,
              est.shape = shapes)
    return(as.data.frame(out))
}


gpdSeqTestsWithFallback <- function(var, thresholds, method, nsim) {
    
    fits <- tryCatch({ # Currently not defined
        fits <- gpdSeqTests(var, thresholds = thresholds, method = method, nsim = nsim)
    },
    error = function(e) {
        fits <- gpdBackupSeqTests(var, thresholds)
    })
}


select_gpd_threshold <- function(var, nthresholds = 28, nsim = 5, alpha = 0.05) {
    #' Selects the optimal threshold for GPD fitting using sequential goodness-of-fit tests.
    #' The function computes a sequence of thresholds based on quantiles of the input data, fits a GPD to the 
    #' exceedances above each threshold, and performs goodness-of-fit tests. The lowest threshold that passes 
    #' the ForwardStop criterion is selected.
    #' 
    #' @param var Numeric vector of data to fit.
    #' @param nthresholds Number of thresholds to test (default: 28).
    #' @param nsim Number of simulations for the goodness-of-fit test (default: 5).
    #' @param alpha Significance level for the ForwardStop criterion (default: 0.05).
    #' @return A list containing the selected threshold, GPD parameters, p-value,
    #' ForwardStop value, and number of exceedances.
    #' 
    #' 

    # Create candidate thresholds based on quantiles of the data
    thresholds <- quantile(var, probs = seq(0.7, 0.98, length.out = nthresholds))

    # Runs sequential goodness-of-fit tests for each threshold and fits GPD
    # Will first try gpdSeqTests, and if it fails, will fall back to gpdBackupSeqTests
    fits <- gpdSeqTestsWithFallback(var, thresholds, method = "ad", nsim = nsim)

    # Return the index of the lowest threshold that passes the ForwardStop criterion
    valid_pk <- fits$ForwardStop
    k    <- min(which(valid_pk > alpha)); # lowest index being "accepted"
    # If no thresholds pass, throw an error and set k to 1 (the lowest threshold)
    if (!is.finite(k)) {
        stop("All thresholds rejected under H0:X~GPD with α=0.05")
        k <- 1
    }
  
    return(list(
        k        = k,
        thresh   = fits$threshold[k],
        theta    = c(fits$est.scale[k], fits$est.shape[k]),
        p.value  = fits$p.value[k],
        pk       = fits$ForwardStop[k],
        n_exceed = fits$num.above[k]
    ))
}


select_weibull_threshold <- function(var, alpha = 0.05, nthresholds = 50) {
    loglikelihood <- function(params, data) {
        # Calculate negative Weibull log-likelihood.
        shape <- params[1]
        scale <- params[2]
        -sum(dweibull(data, shape = shape, scale = scale, log = TRUE))
    }
    ad_test <- function(x, shape, scale, eps=0.05){
        cdf <- function(x) pweibull(x, shape = shape, scale = scale)
        result <- ad.test(x, cdf)
        return(list(p.value = result$p.value))
    }
  
    thresholds <- quantile(var, probs = seq(0.7, 0.98, length.out = nthresholds))
  
    shapes    <- vector(length = length(thresholds))
    scales    <- vector(length = length(thresholds))
    num_above <- vector(length = length(thresholds))
    pvals     <- vector(length = length(thresholds))

    for (i in seq_along(thresholds)) {
        q <- thresholds[i]
        exceedances <- var[var > q] - q

        # initial estimates
        mean_exc <- mean(exceedances)
        init_shape <- 2    # like Rayleigh distribution for winds
        init_scale <- mean_exc

        # fit MLE
        fit <- optim(c(init_shape, init_scale),
                 loglikelihood,
                 data = exceedances,
                 method = "L-BFGS-B",
                 lower = c(0.1, 0.1),
                 upper = c(2, 10))
    
        shapes[i]      <- fit$par[1]
        scales[i]      <- fit$par[2]
        num_above[i]   <- length(exceedances)
        pvals[i]       <- ad_test(exceedances, shapes[i], scales[i])$p.value
    }

    # https://doi.org/10.1214/17-AOAS1092
    pk <- rev(eva:::pSeqStop(rev(pvals))$ForwardStop)
    # m   <- length(pvals)
    # int <- seq(1, m, 1)
    # pk  <- cumsum(-log(1 - pvals[int]))/int
  
    k   <- min(which(pk > alpha)); # lowest index being "accepted"
    if (!is.finite(k)) {
        stop("All thresholds rejected under H0:X~Weibull with α=0.05")
        k <- 1
    }
  
    thresh <- thresholds[k]
    shape  <- shapes[k]
    scale  <- scales[k]
    pval   <- pvals[k]
    pk     <- pk[k]
    exceedances <- var[var > thresh] - thresh
    num_above   <- length(exceedances)

    return(list(
        thresh = thresh,
        theta = c(scale, shape),
        p.value = pval,
        pk = pk,
        n_exceed = num_above
    ))
}


process_gridcell_marginal <- function(gridcell, var, threshold_selector, cdf,
                                      empirical_only = FALSE, test_years = TEST.YEARS) {
    
    # Get the unique grid ID for logging purposes
    grid_id <- unique(gridcell$grid)
    
    # Print a message indicating which grid and variable are being processed
    message(
        "[", format(Sys.time(), "%H:%M:%S"), "] ",
        "Processing grid ", grid_id,
        " for variable ", var
    )

    # Create a data frame to hold the maxima for this grid cell
    # Maxima based on the risk functional for that grid cell
    # For max_temp maxima$variable is copied from gridcell$max_temp
    maxima <- data.frame(
        event_id = gridcell$event_id,
        variable = gridcell[[var]], # current variable of interest
        time = as.POSIXct(                                  # Convert event_start to datetime
            trimws(as.character(gridcell$event_start)),
            format = "%Y-%m-%d %H:%M:%S",
            tz = "UTC"
        ),
        event.rp = gridcell$event_rp,
        grid = gridcell$grid,
        lat = gridcell$lat,
        lon = gridcell$lon
    )

    # Remove rows with non-finite values in the 'variable' column 
    # Should clear computation over oceans when masked out
    maxima <- maxima[is.finite(maxima$variable), ]
    # If there are no valid maxima, return NULL
    if (nrow(maxima) == 0) return(NULL)


    # ---------------------------------------------------------------------------
    # 2 CASES: BOTH TO REMOVE HOLDOUT YEARS FROM TRAINING DATA
    # 1. If the gridcell has an "event_year" column, use it to filter the training data.
    # 2. If not, use the year extracted from the 'time' column of maxima
    if ("event_year" %in% names(gridcell)) {
        event_year <- gridcell$event_year[is.finite(gridcell[[var]])]
        train <- maxima[event_year %ni% test_years, ]
    } else {
        train <- maxima[year(maxima$time) %ni% test_years, ]
    }
    # ----------------------------------------------------------------------------

    # Remove rows with non-finite values in the 'variable' column from the training data
    train <- train[is.finite(train$variable), ]

    # If there are fewer than 5 training samples, cannot fit
    if (nrow(train) < 5) {
        message(
            "[", format(Sys.time(), "%H:%M:%S"), "] ",
            "Grid ", grid_id, ": Not enough training data (n < 5) for variable ", var,
            ". Skipping this grid cell."
        )
        maxima$thresh <- NA
        maxima$scale <- NA
        maxima$shape <- NA
        maxima$p <- 0
        maxima$pk <- 0


        # But can still fit empirical CDF if there is at least 1 training sample
        if (nrow(train) >= 1) {
            maxima$ecdf <- ecdf(train$variable)(maxima$variable)
        } else {
            maxima$ecdf <- NA
        }

        maxima$scdf <- maxima$ecdf
        maxima$box.test <- NA
        return(maxima)
    }
    

    # ----------------------------------------------------------------------------
    # If empirical_only is TRUE and the variable is "num_events", skip GPD fitting

    if (empirical_only && var == "num_events") {
        maxima$thresh <- NA
        maxima$scale <- NA
        maxima$shape <- NA
        maxima$p <- NA
        maxima$pk <- NA

        if (nrow(train) >= 1) {
            trans <- hurdle_ecdf(train$variable)
            maxima$ecdf <- trans(maxima$variable)
        } else {
            maxima$ecdf <- NA
        }

        maxima$scdf <- maxima$ecdf
        maxima$box.test <- NA
        
        return(maxima)
    }

    # ---------------------------------------------------------------------------
    

    tryCatch({

        fit <- threshold_selector(train$variable) # Defined by argument, 
                                                  # e.g., select_gpd_threshold or select_weibull_threshold

        # Extract the fitted parameters and goodness-of-fit statistics
        thresh <- fit$thresh
        scale  <- fit$theta[1]
        shape  <- fit$theta[2]
        pval   <- fit$p.value
        pk     <- fit$pk

        # Store the results in the maxima data frame
        # So every row in a given gridcell gets the same fitted params
        maxima$thresh <- thresh
        maxima$scale  <- scale
        maxima$shape  <- shape
        maxima$p      <- pval
        maxima$pk     <- pk

        # Compute the semi-parametric CDF for the maxima using the fitted GPD parameters
        # Since cdf here is pgpd
        maxima$scdf <- scdf(
            train = train$variable,
            loc = thresh,
            scale = scale,
            shape = shape,
            cdf = cdf
        )(maxima$variable)

        # Compute the empirical CDF still for the maxima based on the training data
        # Just for comparison, not used in the GAN training if scdf is good
        maxima$ecdf <- ecdf(train$variable)(maxima$variable)


        excesses <- maxima$variable[maxima$variable > thresh]
        maxima$box.test <- if (length(excesses) > 1) {
            Box.test(excesses)[["p.value"]]
        } else {
            NA
        }

        # Return the processed maxima for this grid cell
        maxima

    }, error = function(e) { # IF THE FITTING FAILS, RETURN NA FOR THE FITTED PARAMETERS AND CDFs
                             # Then just return ecdf for the maxima based on the training data

        maxima$thresh <- NA
        maxima$scale  <- NA
        maxima$shape  <- NA
        maxima$p      <- 0
        maxima$pk     <- 0
        maxima$ecdf   <- ecdf(train$variable)(maxima$variable)
        maxima$scdf   <- maxima$ecdf
        maxima$box.test <- NA

        maxima
    })
}

process_gridchunk_marginal <- function(gridchunk, var, threshold_selector, cdf,
                                       empirical_only = FALSE, test_years = TEST.YEARS) {

    # Check "grid" column exists
    if (!"grid" %in% names(gridchunk)) {
        stop("gridchunk has no `grid` column. Columns are: ",
                 paste(names(gridchunk), collapse = ", "))
    }

    # Split the gridchunk by individual grid cells for processing
    grid_split <- split(gridchunk, gridchunk[["grid"]])

    # Process each grid cell
    maxima <- lapply(grid_split, function(gridcell) {
        process_gridcell_marginal(
            gridcell = gridcell,
            var = var,
            threshold_selector = threshold_selector,
            cdf = cdf,
            empirical_only = empirical_only,
            test_years = test_years
        )
    })

    # Combine the results back into a single data frame (for that chunk)
    bind_rows(maxima)
}


marginal_transformer <- function(df, threshold_selector, var, q = NA, cdf = NULL,
                                 chunksize = 128, empirical_only = FALSE) {

    # Get the unique grid cells in the data frame
    gridcells <- unique(df$grid)

    # Split the grid cells into chunks of size `chunksize`
    gridchunks <- split(gridcells, ceiling(seq_along(gridcells) / chunksize))
    gridchunks <- unname(gridchunks)

    # Determine the number of workers to use for parallel processing
    nchunks <- length(gridchunks)

    # Set up parallel processing
    # We use `max(1, min(...))` to ensure at least one worker is used and not exceed available cores
    workers <- max(1, min(future::availableCores() - 4, nchunks))
    plan(multisession, workers = workers)

    # One dataframe per chunk
    chunk_data <- lapply(gridchunks, function(g) {
        out <- df[df$grid %in% g, , drop = FALSE]

        if (!"grid" %in% names(out)) {
            stop("Chunk lost grid column")
        }
        out
    })

    # Process chunks in parallel and dataframe them back together
    transformed <- future_map_dfr(    
            chunk_data, # For each chunk:
        function(chunk_df) { 
            process_gridchunk_marginal( # apply the marginal processing function
                gridchunk = chunk_df,
                var = var,
                threshold_selector = threshold_selector,
                cdf = cdf,
                empirical_only = empirical_only,
                test_years = TEST.YEARS
            )
        },
        .options = furrr_options(seed = TRUE))

    # Keep only the relevant columns for the output
    fields <- c(
        "event_id", "variable", "time", "event.rp", "grid", "lat", "lon",
        "thresh", "scale", "shape", "p", "pk", "ecdf", "scdf", "box.test"
    )

    transformed <- transformed[, fields]
    transformed
}


# ----------------------------
# ----------------------------
# ----------------------------


gpd_transformer <- function(df, var, q, chunksize = 256) {
    marginal_transformer(
        df = df,
        threshold_selector = select_gpd_threshold,
        var = var,
        q = q,
        cdf = pgpd,
        chunksize = chunksize
    )
}

num_event_transformer <- function(df, var, chunksize = 256) {
    marginal_transformer(
        df = df,
        threshold_selector = NULL,
        var = var,
        q = NA,
        cdf = NULL,
        chunksize = chunksize,
        empirical_only = TRUE
    )
}

empirical_transformer <- function(df, var, chunksize = 256) {
    marginal_transformer(
        df = df,
        threshold_selector = NULL,
        var = var,
        q = NA,
        cdf = NULL,
        chunksize = chunksize,
        empirical_only = TRUE
    )
}

# ----------------------------
# ----------------------------
# ----------------------------




weibull_transformer <- function(df, metadata, var, q, chunksize = 256) {
    marginal_transformer(
        df = df,
        threshold_selector = select_weibull_threshold,
        metadata = metadata,
        var = var,
        q = q,
        cdf = pweibull,
        chunksize = chunksize
    )
}