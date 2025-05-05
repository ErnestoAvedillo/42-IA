#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <test_part> <test_number[1-5]>"
    echo "1 - visualize data [1-2]"
    echo "2 - Mandatory test [1-5]"
    echo "3 - Bonus test"
    echo "4 - Analysis test [1-2]"
    echo "example: $0 1 2 for test 2 of part 1"
    exit 1
fi

if [ "$1" == "1" ]; then
# Visualize functions
    if [ -z "$2" ]; then
        echo "Usage: $0 1 <test_number[1-5]>"
        echo "example: $0 1 2 for test 2 of part 1"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running visualization for real versus imagined on left and right hand"
        python3 -m Code.Mandatory.visualize "[1]" "[3,4]"
    elif [ "$2" == "2" ]; then
        echo "Running visualization for real versus imagined on both fists or feet"
        python3 -m Code.Mandatory.visualize "[1]" "[5,6]"
    fi
elif [ "$1" == "2" ]; then
# Mandatory tests
    if [ -z "$2" ]; then
        echo "Usage: $0 2 <test_number[1-5]>"
        echo "example: $0 2 2 for test 2 of part 2"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running mandatory test 1 for left versus right hand for one subject"
        python3 -m Code.Mandatory.mybci "[1]" "[3,7,11]" "KNN"
        python3 -m Code.Mandatory.mybci_predict "[1]" "[3]"
    elif [ "$2" == "2" ]; then
        echo "Running mandatory test 1 for imaginary left versus right hand for one subjects"
        python3 -m Code.Mandatory.mybci "[1]" "[4,8,12]" "KNN"
        python3 -m Code.Mandatory.mybci_predict "[1]" "[4]"
    elif [ "$2" == "3" ]; then
        echo "Running mandatory test 1 for both hands versus feet for one subject"
        python3 -m Code.Mandatory.mybci "[1]" "[5,9,13]" "KNN"
        python3 -m Code.Mandatory.mybci_predict "[1]" "[5]"
    elif [ "$2" == "4" ]; then
        echo "Running mandatory test 1 for imaginary both hands versus feet for one subject"
        python3 -m Code.Mandatory.mybci "[1]" "[6,10,14]" "KNN"
        python3 -m Code.Mandatory.mybci_predict "[1]" "[6]"
    elif [ "$2" == "5" ]; then
        echo "Running mandatory test 1 for left versus right hand for several subjects"
        python3 -m Code.Mandatory.mybci "[1,2,3,4]" "[3,7,11]" "KNN"
        python3 -m Code.Mandatory.mybci_predict "[2]" "[3]"
    elif [ "$2" == "6" ]; then
        echo "Running mandatory test 1 for imaginary left versus right hand for several subjects"
        python3 -m Code.Mandatory.mybci "[1,2,3,4]" "[4,8,12]" "KNN"
        python3 -m Code.Mandatory.mybci_predict "[2]" "[4]"
    elif [ "$2" == "7" ]; then
        echo "Running mandatory test 1 for both hands versus feet for several subjects"
        python3 -m Code.Mandatory.mybci "[1,2,3,4]" "[5,9,13]" "KNN"
        python3 -m Code.Mandatory.mybci_predict "[2]" "[5]"
    elif [ "$2" == "8" ]; then
        echo "Running mandatory test 1 for imaginary both hands versus feet for several subjects"
        python3 -m Code.Mandatory.mybci "[1,2,3,4]" "[6,10,14]" "KNN" 
        python3 -m Code.Mandatory.mybci_predict "[2]" "[6]"
    fi
elif [ "$1" == "3" ]; then
# Bonus tests

   if [ -z "$2" ]; then
        echo "Usage: $0 2 <test_number[1-8]>"
        echo "example: $0 2 2 for test 2 of part 2"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running bonus test 1 for left versus right hand for one subject"
        python3 -m Code.Bonus.mybci "[1,2]" "[3,7,11]" "csp"
        python3 -m Code.Bonus.mybci_predict "[1]" "[3]"
    elif [ "$2" == "2" ]; then
        echo "Running bonus test 1 for imaginary left versus right hand for one subjects"
        python3 -m Code.Bonus.mybci "[1]" "[4,8,12]" "csp"
        python3 -m Code.Bonus.mybci_predict "[1]" "[4]"
    elif [ "$2" == "3" ]; then
        echo "Running bonus test 1 for both hands versus feet for one subject"
        python3 -m Code.Bonus.mybci "[1]" "[5,9,13]" "cov"
        python3 -m Code.Bonus.mybci_predict "[1]" "[5]"
    elif [ "$2" == "4" ]; then
        echo "Running bonus test 1 for imaginary both hands versus feet for one subject"
        python3 -m Code.Bonus.mybci "[1]" "[6,10,14]" "csp"
        python3 -m Code.Bonus.mybci_predict "[1]" "[6]"
    elif [ "$2" == "5" ]; then
        echo "Running bonus test 1 for left versus right hand for several subjects"
        python3 -m Code.Bonus.mybci "[1,2,3,4]" "[3,7,11]" "csp"
        python3 -m Code.Bonus.mybci_predict "[2]" "[3]"
    elif [ "$2" == "6" ]; then
        echo "Running bonus test 1 for imaginary left versus right hand for several subjects"
        python3 -m Code.Bonus.mybci "[1,2,3,4]" "[4,8,12]" "csp"
        python3 -m Code.Bonus.mybci_predict "[2]" "[4]"
    elif [ "$2" == "7" ]; then
        echo "Running bonus test 1 for both hands versus feet for several subjects"
        python3 -m Code.Bonus.mybci "[1,2,3,4]" "[5,9,13]" "csp"
        python3 -m Code.Bonus.mybci_predict "[2]" "[5]"
    elif [ "$2" == "8" ]; then
        echo "Running bonus test 1 for imaginary both hands versus feet for several subjects"
        python3 -m Code.Bonus.mybci "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]" "[3,5,7,9,11,13]" "csp"
        python3 -m Code.Bonus.mybci_predict "[2]" "[6]"
    elif [ "$2" == "9" ]; then
        echo "Running bonus test 1 for imaginary both hands versus feet for several subjects"
        python3 -m Code.Bonus.mybci "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]" "[4,6,8,10,12,14]" "cov"
        python3 -m Code.Bonus.mybci_predict "[2]" "[6]"
    fi
elif [ "$1" == "4" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 4 <test_number[1-2]>"
        echo "example: $0 4 2 for test 2 of part 4"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running analysis test 1 for left versus right hand for one subject"
        python3 -m Code.Analysis.mybci "[1]" "[3,7,11]"
    elif [ "$2" == "2" ]; then
        echo "Running analysis test 1 for imaginary left versus right hand for one subjects"
        python3 -m Code.Analysis.mybci "[1]" "[4,8,12]" 
    elif [ "$2" == "3" ]; then
        echo "Running analysis test 1 for both hands versus feet for one subject"
        python3 -m Code.Analysis.mybci "[1]" "[5,9,13]" 
    elif [ "$2" == "4" ]; then
        echo "Running analysis test 1 for imaginary both hands versus feet for one subject"
        python3 -m Code.Analysis.mybci "[1]" "[6,10,14]" 
    elif [ "$2" == "5" ]; then
        echo "Running analysis test 1 for left versus right hand for several subjects"
        python3 -m Code.Analysis.mybci "[1,2,3,4]" "[3,7,11]"
    elif [ "$2" == "6" ]; then
        echo "Running analysis test 1 for left versus right hand real and imaginary for several subjects"
        python3 -m Code.Analysis.mybci "[1,2,3,4,5,6,7,8]" "[3,5,7,9,11,13]"
    fi
elif  [ "$1" == "5" ]; then
# Bonus tests alternative
    if [ -z "$2" ]; then
        echo "Usage: $0 4 <test_number[1-2]>"
        echo "example: $0 4 2 for test 2 of part 4"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running bonus test 5 for left versus right hand for one subject"
        python3 -m Code.Bonus.mybci_std "[1]" "[3,5,7,9,11,13]" "csp"
        python3 -m Code.Bonus.mybci_predict_std "[1]" "[5]"
    fi
else
    echo "Invalid test part. Use 1, 2,  3 or 4."
    exit 1
fi
