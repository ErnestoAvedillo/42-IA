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
    if [ -z "$2" ]; then
        echo "Usage: $0 1 <test_number[1-5]>"
        echo "example: $0 1 2 for test 2 of part 1"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running visualization for real versus imagined on left and right hand"
        python -m Code.Mandatory.visualize "[1]" "[3,4]"
    elif [ "$2" == "2" ]; then
        echo "Running visualization for real versus imagined on both fists or feet"
        python -m Code.Mandatory.visualize "[1]" "[5,6]"
    fi
elif [ "$1" == "2" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 2 <test_number[1-5]>"
        echo "example: $0 2 2 for test 2 of part 2"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running mandatory test 1 for left versus right hand for one subject"
        python -m Code.Mandatory.mybci "[1]" "[3,7,11]" "KNN"
    elif [ "$2" == "2" ]; then
        echo "Running mandatory test 1 for imaginary left versus right hand for one subjects"
        python -m Code.Mandatory.mybci "[1]" "[4,8,12]" "KNN"
    elif [ "$2" == "3" ]; then
        echo "Running mandatory test 1 for both hands versus feet for one subject"
        python -m Code.Mandatory.mybci "[1]" "[5,9,13]" "KNN" --plot
    elif [ "$2" == "4" ]; then
        echo "Running mandatory test 1 for imaginary both hands versus feet for one subject"
        python -m Code.Mandatory.mybci "[1]" "[6,10,14]" "KNN" --plot
    elif [ "$2" == "5" ]; then
        echo "Running mandatory test 1 for left versus right hand for several subjects"
        python -m Code.Mandatory.mybci "[1,2,3,4]" "[3,7,11]" "KNN"
    elif [ "$2" == "6" ]; then
        echo "Running mandatory test 1 for imaginary left versus right hand for several subjects"
        python -m Code.Mandatory.mybci "[1,2,3,4]" "[4,8,12]" "KNN"
    elif [ "$2" == "7" ]; then
        echo "Running mandatory test 1 for both hands versus feet for several subjects"
        python -m Code.Mandatory.mybci "[1,2,3,4]" "[5,9,13]" "KNN" --plot
    elif [ "$2" == "8" ]; then
        echo "Running mandatory test 1 for imaginary both hands versus feet for several subjects"
        python -m Code.Mandatory.mybci "[1,2,3,4]" "[6,10,14]" "KNN" --plot
    fi
elif [ "$1" == "3" ]; then
   if [ -z "$2" ]; then
        echo "Usage: $0 2 <test_number[1-8]>"
        echo "example: $0 2 2 for test 2 of part 2"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running bonus test 1 for left versus right hand for one subject"
        python -m Code.Bonus.mybci "[1]" "[3,7,11]" "csp"
    elif [ "$2" == "2" ]; then
        echo "Running bonus test 1 for imaginary left versus right hand for one subjects"
        python -m Code.Bonus.mybci "[1]" "[4,8,12]" "csp"
    elif [ "$2" == "3" ]; then
        echo "Running bonus test 1 for both hands versus feet for one subject"
        python -m Code.Bonus.mybci "[1]" "[5,9,13]" "csp" --plot
    elif [ "$2" == "4" ]; then
        echo "Running bonus test 1 for imaginary both hands versus feet for one subject"
        python -m Code.Bonus.mybci "[1]" "[6,10,14]" "csp" --plot
    elif [ "$2" == "5" ]; then
        echo "Running bonus test 1 for left versus right hand for several subjects"
        python -m Code.Bonus.mybci "[1,2,3,4]" "[3,7,11]" "csp"
    elif [ "$2" == "6" ]; then
        echo "Running bonus test 1 for imaginary left versus right hand for several subjects"
        python -m Code.Bonus.mybci "[1,2,3,4]" "[4,8,12]" "csp"
    elif [ "$2" == "7" ]; then
        echo "Running bonus test 1 for both hands versus feet for several subjects"
        python -m Code.Bonus.mybci "[1,2,3,4]" "[5,9,13]" "csp" --plot
    elif [ "$2" == "8" ]; then
        echo "Running bonus test 1 for imaginary both hands versus feet for several subjects"
        python -m Code.Bonus.mybci "[1,2,3,4]" "[6,10,14]" "csp" --plot
    fi
elif [ "$1" == "4" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 4 <test_number[1-2]>"
        echo "example: $0 4 2 for test 2 of part 4"
        exit 1
    fi
    if [ "$2" == "1" ]; then
        echo "Running analysis test 1 for left versus right hand for one subject"
        python -m Code.Analysis.mybci "[1]" "[3,7,11]"
    elif [ "$2" == "2" ]; then
        echo "Running analysis test 1 for imaginary left versus right hand for one subjects"
        python -m Code.Analysis.mybci "[1]" "[4,8,12]" 
    elif [ "$2" == "3" ]; then
        echo "Running analysis test 1 for both hands versus feet for one subject"
        python -m Code.Analysis.mybci "[1]" "[5,9,13]" 
    elif [ "$2" == "4" ]; then
        echo "Running analysis test 1 for imaginary both hands versus feet for one subject"
        python -m Code.Analysis.mybci "[1]" "[6,10,14]" 
    elif [ "$2" == "5" ]; then
        echo "Running analysis test 1 for left versus right hand for several subjects"
        python -m Code.Analysis.mybci "[1,2,3,4]" "[3,7,11]"
    fi
else
    echo "Invalid test part. Use 1, 2,  3 or 4."
    exit 1
fi
