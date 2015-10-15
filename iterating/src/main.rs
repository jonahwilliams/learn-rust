fn main() {

    let mut range = 0..10;

    loop {
        match range.next() {
            Some(x) => {
                println!("{}", x);
            },
            None => { break }
        }
    }

    let nums = vec![1, 2, 3];

    for num in &nums {
        println!("{}", num);
    }

    let one_to_one_hundred = (1..101).collect::<Vec<i32>>();

    let greater_than_forty_two = (0..100).find(|x| *x > 42);

    match greater_than_forty_two {
        Some(_) => println!("We got some numbers!"),
        None => println!("No numbers found!"),
    }

    let sum = (1..4).fold(0, |sum, x| sum + x);
}
